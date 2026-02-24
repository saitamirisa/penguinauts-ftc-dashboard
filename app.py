#!/usr/bin/env python3
import os
import json
import math
import time
from typing import Dict, Any, Tuple, List, Set

import requests
import pandas as pd
import streamlit as st

# ----------------------------
# Constants
# ----------------------------
REST_BASE = "https://api.ftcscout.org/rest/v1"
GQL_URL = "https://api.ftcscout.org/graphql"

CACHE_DIR = "./ftc_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

S = requests.Session()

PENGUINAUTS_TEAM = 32240


# ----------------------------
# Simple disk cache helpers (optional)
# ----------------------------
def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def load_cache(name: str):
    p = _cache_path(name)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_cache(name: str, obj):
    p = _cache_path(name)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ----------------------------
# HTTP helpers
# ----------------------------
def get_json(url: str, params=None, retries: int = 4, timeout=(10, 60)):
    for attempt in range(retries):
        try:
            r = S.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404,):
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"GET {url} failed {r.status_code}: {r.text[:200]}")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after retries")


def gql(query: str, variables: dict, retries: int = 4, timeout=(10, 90)):
    for attempt in range(retries):
        try:
            r = S.post(GQL_URL, json={"query": query, "variables": variables}, timeout=timeout)
            if r.status_code == 200:
                payload = r.json()
                if "errors" in payload:
                    raise RuntimeError(json.dumps(payload["errors"], indent=2))
                return payload["data"]
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"POST GraphQL failed {r.status_code}: {r.text[:200]}")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("GraphQL failed after retries")


# ----------------------------
# Scoring helpers
# ----------------------------
def minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([math.nan] * len(s), index=s.index)
    norm = (s - mn) / (mx - mn)
    return norm if higher_is_better else (1.0 - norm)


def confidence_factor(n_matches: pd.Series) -> pd.Series:
    n = pd.to_numeric(n_matches, errors="coerce").fillna(0.0)
    # approaches 1.0 as matches increase; small boost for more data
    return 0.85 + 0.15 * (1.0 - (1.0 / (1.0 + n / 12.0)))


def weights_ui(prefix: str, title: str, defaults: Dict[str, int], allow_pen_zero: bool = True) -> Dict[str, Any]:
    st.subheader(title)
    cols = st.columns(5)

    np_w = cols[0].slider("NP", 1, 100, int(defaults.get("np", 25)), step=1, key=f"{prefix}_np")
    auto_w = cols[1].slider("Auto", 1, 100, int(defaults.get("auto", 25)), step=1, key=f"{prefix}_auto")
    tele_w = cols[2].slider("TeleOp", 1, 100, int(defaults.get("tele", 25)), step=1, key=f"{prefix}_tele")
    eg_w = cols[3].slider("Endgame", 1, 100, int(defaults.get("eg", 25)), step=1, key=f"{prefix}_eg")

    if allow_pen_zero:
        pen_min = 0
    else:
        pen_min = 1
    pen_w = cols[4].slider("Penalty", pen_min, 100, int(defaults.get("pen", 0)), step=1, key=f"{prefix}_pen")

    total = np_w + auto_w + tele_w + eg_w + pen_w
    if total <= 0:
        total = 1

    w = {
        "np": np_w / total,
        "auto": auto_w / total,
        "tele": tele_w / total,
        "eg": eg_w / total,
        "pen": pen_w / total,
        "raw": {"np": np_w, "auto": auto_w, "tele": tele_w, "eg": eg_w, "pen": pen_w},
    }
    st.caption(
        f"Normalized: NP {w['np']:.2f}, Auto {w['auto']:.2f}, TeleOp {w['tele']:.2f}, "
        f"Endgame {w['eg']:.2f}, Pen {w['pen']:.2f}"
    )
    return w


def score_dataframe(df_base: pd.DataFrame, w_perf: Dict[str, float], w_fit: Dict[str, float]) -> pd.DataFrame:
    """
    df_base must already have:
      np_norm, auto_norm, teleop_norm, eg_norm, pen_norm, season_matches
    Produces scout_score and alliance_fit_score as whole-number %.
    """
    df = df_base.copy()

    perf = (
        w_perf["np"] * df["np_norm"]
        + w_perf["auto"] * df["auto_norm"]
        + w_perf["tele"] * df["teleop_norm"]
        + w_perf["eg"] * df["eg_norm"]
        + w_perf["pen"] * df["pen_norm"]
    )
    scout = perf * confidence_factor(df["season_matches"])
    df["scout_score"] = (pd.to_numeric(scout, errors="coerce") * 100).round(0).astype("Int64")

    fit = (
        w_fit["auto"] * df["auto_norm"]
        + w_fit["eg"] * df["eg_norm"]
        + w_fit["pen"] * df["pen_norm"]
        + w_fit["tele"] * df["teleop_norm"]
    )
    df["alliance_fit_score"] = (pd.to_numeric(fit, errors="coerce") * 100).round(0).astype("Int64")
    return df


# ----------------------------
# FTCScout data fetch
# ----------------------------
TEAM_MATCHRECORDS_2025 = """
query MatchRecords($season: Int!, $skip: Int!, $take: Int!, $region: RegionOption) {
  matchRecords(season: $season, skip: $skip, take: $take, region: $region) {
    count
    data {
      data {
        alliance
        match {
          hasBeenPlayed
          scores {
            ... on MatchScores2025 {
              red { totalPointsNp penaltyPointsCommitted }
              blue { totalPointsNp penaltyPointsCommitted }
            }
          }
          teams {
            teamNumber
            alliance
            dq
            noShow
            onField
          }
        }
      }
    }
  }
}
"""


@st.cache_data(ttl=60 * 60 * 12, show_spinner=True)  # 12 hours
def get_season_avgs_cached(season: int, region: str) -> Dict[int, Dict[str, Any]]:
    """
    Expensive region-wide scan. Cached for 12 hours by Streamlit.
    Returns: {teamNumber: {season_matches, season_avg_np, season_avg_pen}}
    """
    agg = {}
    page_size = 300
    sleep_s = 0.02

    first = gql(TEAM_MATCHRECORDS_2025, {"season": season, "skip": 0, "take": page_size, "region": region})
    mr = first["matchRecords"]
    total = mr["count"]

    def process_rows(rows):
        for row in rows:
            alliance = row["data"]["alliance"]
            match = row["data"]["match"]
            if not match.get("hasBeenPlayed"):
                continue
            scores = match.get("scores") or {}

            if alliance == "Red":
                a = scores.get("red") or {}
            elif alliance == "Blue":
                a = scores.get("blue") or {}
            else:
                continue

            np_val = a.get("totalPointsNp")
            pen_val = a.get("penaltyPointsCommitted")
            if np_val is None:
                continue

            for p in (match.get("teams") or []):
                if p.get("alliance") != alliance:
                    continue
                if not p.get("onField"):
                    continue
                if p.get("noShow") or p.get("dq"):
                    continue

                t = p.get("teamNumber")
                if t is None:
                    continue

                d = agg.setdefault(t, {"np_sum": 0.0, "np_count": 0, "pen_sum": 0.0, "pen_count": 0})
                d["np_sum"] += float(np_val)
                d["np_count"] += 1
                if pen_val is not None:
                    d["pen_sum"] += float(pen_val)
                    d["pen_count"] += 1

    process_rows(mr["data"])
    skip = page_size
    while skip < total:
        page = gql(TEAM_MATCHRECORDS_2025, {"season": season, "skip": skip, "take": page_size, "region": region})
        mr2 = page["matchRecords"]
        process_rows(mr2["data"])
        skip += page_size
        time.sleep(sleep_s)

    season_avgs = {}
    for t, d in agg.items():
        season_avgs[int(t)] = {
            "season_matches": d["np_count"],
            "season_avg_np": (d["np_sum"] / d["np_count"]) if d["np_count"] else None,
            "season_avg_pen": (d["pen_sum"] / d["pen_count"]) if d["pen_count"] else None,
        }
    return season_avgs


@st.cache_data(ttl=60 * 60 * 6, show_spinner=True)  # 6 hours
def fetch_event_roster(season: int, event_code: str) -> Tuple[Dict[str, Any], List[int]]:
    """
    Uses REST event + REST event teams roster (roster includes some inactive entries sometimes).
    """
    event = get_json(f"{REST_BASE}/events/{season}/{event_code}") or {}
    teps = get_json(f"{REST_BASE}/events/{season}/{event_code}/teams") or []
    roster = sorted({int(t["teamNumber"]) for t in teps if t.get("teamNumber") is not None})
    return event, roster


@st.cache_data(ttl=60, show_spinner=True)  # 60 seconds, event live data
def fetch_event_live_np_pen_and_active(season: int, event_code: str) -> Tuple[Dict[int, Dict[str, Any]], Set[int], bool]:
    """
    Uses REST event matches to compute EVENT averages and active teams.
    Returns (event_avgs_by_team, active_team_set, any_played_matches)
    """
    matches = get_json(f"{REST_BASE}/events/{season}/{event_code}/matches") or []
    agg = {}
    active = set()
    any_played = False

    for m in matches:
        # If scores missing, match likely unplayed
        scores = m.get("scores")
        if not scores:
            continue
        any_played = True

        # Scores schema depends on season; for 2025 we expect totalPointsNp + penaltyPointsCommitted in alliance objects.
        # FTCScout REST usually returns fields similar to GraphQL. We'll be defensive.
        red = (scores.get("red") or {})
        blue = (scores.get("blue") or {})

        # teams list typically in m["teams"]
        teams = m.get("teams") or []

        def add_for_alliance(alliance_name: str, alliance_scores: Dict[str, Any]):
            np_val = alliance_scores.get("totalPointsNp")
            pen_val = alliance_scores.get("penaltyPointsCommitted")
            if np_val is None:
                return

            for p in teams:
                if p.get("alliance") != alliance_name:
                    continue
                if not p.get("onField", True):
                    continue
                if p.get("noShow") or p.get("dq"):
                    continue
                t = p.get("teamNumber")
                if t is None:
                    continue
                t = int(t)
                active.add(t)
                d = agg.setdefault(t, {"np_sum": 0.0, "np_count": 0, "pen_sum": 0.0, "pen_count": 0})
                d["np_sum"] += float(np_val)
                d["np_count"] += 1
                if pen_val is not None:
                    d["pen_sum"] += float(pen_val)
                    d["pen_count"] += 1

        add_for_alliance("Red", red)
        add_for_alliance("Blue", blue)

    event_avgs = {}
    for t, d in agg.items():
        event_avgs[int(t)] = {
            "event_matches": d["np_count"],
            "event_avg_np": (d["np_sum"] / d["np_count"]) if d["np_count"] else None,
            "event_avg_pen": (d["pen_sum"] / d["pen_count"]) if d["pen_count"] else None,
        }

    return event_avgs, active, any_played


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_team_cached(team_number: int) -> Dict[str, Any]:
    team = get_json(f"{REST_BASE}/teams/{team_number}") or {}
    return team


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def safe_quickstats(team_number: int, season: int, region: str) -> Dict[str, Any]:
    """
    Some teams will 404 quick-stats if they have no events in that season/region.
    Return {} instead of raising.
    """
    qs = get_json(f"{REST_BASE}/teams/{team_number}/quick-stats", params={"season": season, "region": region})
    return qs or {}


def build_dataframe(
    season: int,
    event_code: str,
    region: str,
    mode: str,
    include_inactive_override: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame, bool, int, int]:
    # 1) roster
    event, roster = fetch_event_roster(season, event_code)

    # 2) event-only NP/pen + active teams
    event_avgs, active, any_played = fetch_event_live_np_pen_and_active(season, event_code)

    # Mode logic
    if mode == "Pre-Game Analysis":
        teams = roster
        active_filtering_used = False
    else:
        if include_inactive_override:
            teams = roster
            active_filtering_used = False
        else:
            if not any_played:
                teams = roster
                active_filtering_used = False
            else:
                active_list = [t for t in roster if t in active]
                if len(active_list) == 0:
                    teams = roster
                    active_filtering_used = False
                else:
                    teams = active_list
                    active_filtering_used = True

    # 3) season averages (cached)
    season_avgs = get_season_avgs_cached(season, region)

    rows = []
    for t in teams:
        qs = safe_quickstats(t, season, region)

        def stat(qs_obj, key):
            s = (qs_obj or {}).get(key) or {}
            return s.get("value"), s.get("rank")

        tot_v, tot_r = stat(qs, "tot")
        auto_v, _ = stat(qs, "auto")
        tele_v, _ = stat(qs, "dc")
        eg_v, _ = stat(qs, "eg")

        s = season_avgs.get(int(t), {})
        e = event_avgs.get(int(t), {})

        team_obj = fetch_team_cached(int(t))
        name = team_obj.get("name") or f"Team {t}"

        rows.append({
            "team_number": int(t),
            "team_name": name,
            "active_today": (int(t) in active),

            "total_value": tot_v,
            "total_rank": tot_r,
            "auto_value": auto_v,
            "teleop_value": tele_v,
            "endgame_value": eg_v,

            "season_matches": s.get("season_matches", 0),
            "season_avg_np": s.get("season_avg_np"),
            "season_avg_pen": s.get("season_avg_pen"),

            "event_matches": e.get("event_matches", 0),
            "event_avg_np": e.get("event_avg_np"),
            "event_avg_pen": e.get("event_avg_pen"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return event, df, active_filtering_used, len(active), len(roster)

    # normalize on season baseline
    df["np_norm"] = minmax(df["season_avg_np"], True)
    df["auto_norm"] = minmax(df["auto_value"], True)
    df["teleop_norm"] = minmax(df["teleop_value"], True)
    df["eg_norm"] = minmax(df["endgame_value"], True)
    df["pen_norm"] = minmax(df["season_avg_pen"], False)

    # round numeric display (raw columns) here; scores are computed per-tab
    for c in [
        "total_value", "auto_value", "teleop_value", "endgame_value",
        "season_avg_np", "season_avg_pen",
        "event_avg_np", "event_avg_pen"
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(1)

    return event, df, active_filtering_used, len(active), len(roster)


# ----------------------------
# UI helpers: pick list interactions
# ----------------------------
def apply_removed_teams(df: pd.DataFrame) -> pd.DataFrame:
    removed = st.session_state.get("removed_teams", set())
    if not removed:
        return df
    return df[~df["team_number"].isin(list(removed))].copy()


def pick_list_controls(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """
    Allows user to remove teams from pick list via multiselect.
    """
    removed = st.session_state.setdefault("removed_teams", set())

    st.markdown("### Pick List Controls")
    cols = st.columns([2, 1, 1])

    # select to remove
    options = df["team_number"].tolist()
    to_remove = cols[0].multiselect(
        "Remove teams from pick list",
        options=options,
        format_func=lambda x: f"{x} — {df.loc[df['team_number']==x, 'team_name'].iloc[0] if (df['team_number']==x).any() else ''}",
        key=f"{key_prefix}_remove_select"
    )
    if cols[1].button("Remove selected", key=f"{key_prefix}_remove_btn"):
        for t in to_remove:
            removed.add(int(t))
        st.session_state["removed_teams"] = removed
        st.rerun()

    if cols[2].button("Reset removed teams", key=f"{key_prefix}_reset_btn"):
        st.session_state["removed_teams"] = set()
        st.rerun()

    st.caption(f"Removed teams count: {len(st.session_state.get('removed_teams', set()))}")
    return apply_removed_teams(df)


def render_pick_list_tab(
    df_base: pd.DataFrame,
    tab_name: str,
    perf_defaults: Dict[str, int],
    fit_defaults: Dict[str, int],
    sort_by: List[str],
    sort_asc: List[bool],
    key_prefix: str,
):
    st.header(tab_name)

    # Weight tuning per pick list
    w_perf = weights_ui(f"{key_prefix}_perf", "ScoutScore weights", perf_defaults, allow_pen_zero=True)
    st.divider()
    w_fit = weights_ui(f"{key_prefix}_fit", "AllianceFit weights", fit_defaults, allow_pen_zero=True)

    # Compute scores for this tab
    df_scored = score_dataframe(df_base, w_perf, w_fit)

    # Sort
    df_scored = df_scored.sort_values(
        by=sort_by,
        ascending=sort_asc,
        na_position="last"
    ).reset_index(drop=True)

    # Pick list remove controls (works for pre-game and game day)
    df_scored = pick_list_controls(df_scored, key_prefix=key_prefix)

    df_scored = df_scored.reset_index(drop=True)
    df_scored.insert(0, "No.", range(1, len(df_scored) + 1))

    show_cols = [
        "No.",
        "team_number",
        "team_name",
        "scout_score",
        "alliance_fit_score",
        "total_value",
        "total_rank",
        "auto_value",
        "teleop_value",
        "endgame_value",
        "season_avg_np",
        "season_avg_pen",
        "season_matches",
        "active_today",
    ]
    st.dataframe(df_scored[show_cols], use_container_width=True, height=650)


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="FTC Scouting Dashboard", layout="wide")

st.title("FTC Scouting Dashboard")

# Ensure session state keys exist
st.session_state.setdefault("removed_teams", set())
st.session_state.setdefault("force_event_refresh", 0.0)

with st.sidebar:
    st.header("Event Settings")
    season = st.number_input("Season", min_value=2019, max_value=2030, value=2025, step=1)
    event_code = st.text_input("Event code", value="USTXCMP4T")
    region = st.text_input("Region (RegionOption)", value="USTX")

    st.divider()

    mode = st.radio("Mode", ["Pre-Game Analysis", "Game Day (Live)"], index=0)

    include_inactive_override = False
    if mode == "Game Day (Live)":
        include_inactive_override = st.checkbox("Include teams not active today", value=False)

    refresh = st.button("Refresh Now")

    st.divider()
    st.subheader("Pick List")
    if st.button("Reset removed teams"):
        st.session_state.removed_teams = set()
        st.rerun()

# Refresh behavior:
# - For event-live data we rely on st.cache_data TTL=60s.
# - Force a rerun; cache will naturally refresh if TTL elapsed.
if refresh:
    st.session_state["force_event_refresh"] = time.time()
    st.rerun()

# Load dataframe
event, df_base, active_filtering_used, active_count, roster_count = build_dataframe(
    season=season,
    event_code=event_code.strip(),
    region=region.strip(),
    mode=mode,
    include_inactive_override=include_inactive_override,
)

if df_base.empty:
    st.warning(
        "No teams returned. Double-check the event code. "
        "If this is Game Day and matches haven't started yet, switch to Pre-Game Analysis."
    )
    st.stop()

st.caption(
    f"Event: **{event.get('name','(unknown)')}** | Mode: **{mode}** | "
    f"Roster size: {roster_count} | Active today: {active_count} | Showing: {len(df_base)} | "
    f"Removed: {len(st.session_state.removed_teams)}"
)

if mode == "Game Day (Live)" and (not include_inactive_override) and (not active_filtering_used):
    st.info("No played matches found yet — showing full roster until matches begin.")

tabs = st.tabs(["Pre-Game (32240)", "Pick List A", "Pick List B", "Pick List C", "Compare Teams", "Team Detail"])


# ----------------------------
# Tab 0: Pre-Game (Penguinauts)
# ----------------------------
with tabs[0]:
    st.header(f"Pre-Game Analysis — Penguinauts {PENGUINAUTS_TEAM} vs this event roster")

    # Use Pick List A defaults for the summary ranking
    w_perf_default = {"np": 32, "auto": 22, "tele": 22, "eg": 22, "pen": 2}
    w_fit_default = {"np": 0, "auto": 43, "tele": 25, "eg": 27, "pen": 5}

    df_scored = score_dataframe(
        df_base,
        {"np": w_perf_default["np"]/100, "auto": w_perf_default["auto"]/100, "tele": w_perf_default["tele"]/100, "eg": w_perf_default["eg"]/100, "pen": w_perf_default["pen"]/100},
        {"auto": w_fit_default["auto"]/100, "tele": w_fit_default["tele"]/100, "eg": w_fit_default["eg"]/100, "pen": w_fit_default["pen"]/100, "np": 0.0},
    )

    df_scored = df_scored.sort_values(
        by=["scout_score", "alliance_fit_score", "total_value", "total_rank"],
        ascending=[False, False, False, True],
        na_position="last"
    ).reset_index(drop=True)
    df_scored.insert(0, "No.", range(1, len(df_scored) + 1))

    # Summary cards for Penguinauts
    if PENGUINAUTS_TEAM in df_scored["team_number"].values:
        idx = int(df_scored.index[df_scored["team_number"] == PENGUINAUTS_TEAM][0])
        row = df_scored.iloc[idx]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("32240 Rank (by ScoutScore)", f"{idx+1}/{len(df_scored)}")
        c2.metric("32240 ScoutScore", f"{row['scout_score']}%")
        c3.metric("32240 AllianceFit", f"{row['alliance_fit_score']}%")
        c4.metric("Total Value / Rank", f"{row['total_value']} / {row['total_rank']}")
    else:
        st.warning(f"Team {PENGUINAUTS_TEAM} not found in this roster.")

    st.subheader("All teams (full roster)")
    show_cols = [
        "No.", "team_number", "team_name",
        "scout_score", "alliance_fit_score",
        "total_value", "total_rank",
        "auto_value", "teleop_value", "endgame_value",
        "season_avg_np", "season_avg_pen", "season_matches",
        "active_today",
    ]
    st.dataframe(df_scored[show_cols], use_container_width=True, height=650)


# ----------------------------
# Pick List A
# ----------------------------
with tabs[1]:
    render_pick_list_tab(
        df_base=df_base,
        tab_name="Pick List A (Balanced)",
        perf_defaults={"np": 32, "auto": 22, "tele": 22, "eg": 22, "pen": 2},
        fit_defaults={"np": 0, "auto": 43, "tele": 25, "eg": 27, "pen": 5},
        sort_by=["scout_score", "alliance_fit_score", "total_value", "total_rank"],
        sort_asc=[False, False, False, True],
        key_prefix="pickA"
    )


# ----------------------------
# Pick List B (Defense / Low penalty risk, but not dominating)
# ----------------------------
with tabs[2]:
    render_pick_list_tab(
        df_base=df_base,
        tab_name="Pick List B (Defense / Low-pen emphasis)",
        perf_defaults={"np": 34, "auto": 20, "tele": 22, "eg": 22, "pen": 2},
        fit_defaults={"np": 0, "auto": 38, "tele": 30, "eg": 27, "pen": 5},
        # sort: your new preference for pick order: ScoutScore -> Fit -> Total -> Rank
        sort_by=["scout_score", "alliance_fit_score", "total_value", "total_rank"],
        sort_asc=[False, False, False, True],
        key_prefix="pickB"
    )


# ----------------------------
# Pick List C (Auto-first / back shooting candidates are NOT directly detectable here)
# ----------------------------
with tabs[3]:
    render_pick_list_tab(
        df_base=df_base,
        tab_name="Pick List C (Auto-first)",
        perf_defaults={"np": 28, "auto": 30, "tele": 20, "eg": 20, "pen": 2},
        fit_defaults={"np": 0, "auto": 50, "tele": 20, "eg": 25, "pen": 5},
        sort_by=["scout_score", "alliance_fit_score", "total_value", "total_rank"],
        sort_asc=[False, False, False, True],
        key_prefix="pickC"
    )


# ----------------------------
# Compare Teams
# ----------------------------
with tabs[4]:
    st.header("Compare Teams")
    a, b = st.columns(2)

    team_list = df_base["team_number"].tolist()
    t1 = a.selectbox("Team 1", team_list, index=0)
    t2 = b.selectbox("Team 2", team_list, index=1 if len(team_list) > 1 else 0)

    def team_row(t):
        r = df_base[df_base["team_number"] == t].iloc[0]
        return r

    r1 = team_row(t1)
    r2 = team_row(t2)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"{t1} — {r1['team_name']}")
        st.json({
            "Total Value": r1["total_value"],
            "Total Rank": r1["total_rank"],
            "Auto": r1["auto_value"],
            "TeleOp": r1["teleop_value"],
            "Endgame": r1["endgame_value"],
            "Season Avg NP": r1["season_avg_np"],
            "Season Avg Pen": r1["season_avg_pen"],
            "Season Matches": r1["season_matches"],
            "Active Today": bool(r1["active_today"]),
        })
    with c2:
        st.subheader(f"{t2} — {r2['team_name']}")
        st.json({
            "Total Value": r2["total_value"],
            "Total Rank": r2["total_rank"],
            "Auto": r2["auto_value"],
            "TeleOp": r2["teleop_value"],
            "Endgame": r2["endgame_value"],
            "Season Avg NP": r2["season_avg_np"],
            "Season Avg Pen": r2["season_avg_pen"],
            "Season Matches": r2["season_matches"],
            "Active Today": bool(r2["active_today"]),
        })

    st.caption("Tip: Pick Lists A/B/C have different tunings and can be adjusted independently.")


# ----------------------------
# Team Detail
# ----------------------------
with tabs[5]:
    st.header("Team Detail")
    t = st.selectbox("Select team", df_base["team_number"].tolist(), index=0)
    row = df_base[df_base["team_number"] == t].iloc[0]

    st.subheader(f"{t} — {row['team_name']}")
    st.write("**Quick overview**")
    st.write(
        f"- Total: {row['total_value']} (Rank {row['total_rank']})\n"
        f"- Auto: {row['auto_value']} | TeleOp: {row['teleop_value']} | Endgame: {row['endgame_value']}\n"
        f"- Season Avg NP: {row['season_avg_np']} | Season Avg Pen: {row['season_avg_pen']} | Matches: {row['season_matches']}\n"
        f"- Active today: {bool(row['active_today'])}"
    )

    # Simple bar chart comparing values (team vs field avg)
    field_avg = {
        "Auto": float(pd.to_numeric(df_base["auto_value"], errors="coerce").mean(skipna=True)),
        "TeleOp": float(pd.to_numeric(df_base["teleop_value"], errors="coerce").mean(skipna=True)),
        "Endgame": float(pd.to_numeric(df_base["endgame_value"], errors="coerce").mean(skipna=True)),
        "NP": float(pd.to_numeric(df_base["season_avg_np"], errors="coerce").mean(skipna=True)),
        "Pen": float(pd.to_numeric(df_base["season_avg_pen"], errors="coerce").mean(skipna=True)),
    }

    team_vals = {
        "Auto": row["auto_value"],
        "TeleOp": row["teleop_value"],
        "Endgame": row["endgame_value"],
        "NP": row["season_avg_np"],
        "Pen": row["season_avg_pen"],
    }

    chart_df = pd.DataFrame({
        "Metric": list(team_vals.keys()) + list(field_avg.keys()),
        "Value": [team_vals[k] for k in team_vals.keys()] + [field_avg[k] for k in field_avg.keys()],
        "Series": ["Team"] * len(team_vals) + ["Field Avg"] * len(field_avg),
    })

    st.bar_chart(chart_df, x="Metric", y="Value", color="Series")
