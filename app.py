#!/usr/bin/env python3
import os
import json
import math
import time
from typing import Dict, Any, Tuple, List, Set, Optional

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
# Auth (Streamlit Google login)
# ----------------------------
def login_screen():
    st.header("This app is private.")
    st.subheader("Please log in.")
    st.button("Log in with Google", on_click=st.login)

# If auth isn't configured, st.user won't have attributes
if not hasattr(st.user, "is_logged_in"):
    st.error("Authentication is not configured for this deployment. Please add the [auth] section to Streamlit secrets.")
    st.stop()

if not st.user.is_logged_in:
    login_screen()
    st.stop()

ALLOWED_EMAILS = {
    "saihero@gmail.com",
    "preetibtv@gmail.com",
}
email = getattr(st.user, "email", None)
if email not in ALLOWED_EMAILS:
    st.error("You are not authorized to use this dashboard.")
    st.button("Log out", on_click=st.logout)
    st.stop()

st.sidebar.success(f"Signed in as {getattr(st.user, 'name', 'User')}")

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
    cols = st.columns(4)

    np_w = cols[0].slider("NP", 1, 100, int(defaults.get("np", 34)), step=1, key=f"{prefix}_np")
    auto_w = cols[1].slider("Auto", 1, 100, int(defaults.get("auto", 22)), step=1, key=f"{prefix}_auto")
    tele_w = cols[2].slider("TeleOp", 1, 100, int(defaults.get("tele", 22)), step=1, key=f"{prefix}_tele")

    pen_min = 0 if allow_pen_zero else 1
    pen_w = cols[3].slider("Penalty", pen_min, 100, int(defaults.get("pen", 2)), step=1, key=f"{prefix}_pen")

    total = np_w + auto_w + tele_w + pen_w
    if total <= 0:
        total = 1

    w = {
        "np": np_w / total,
        "auto": auto_w / total,
        "tele": tele_w / total,
        "pen": pen_w / total,
        "raw": {"np": np_w, "auto": auto_w, "tele": tele_w, "pen": pen_w},
    }
    st.caption(f"Normalized: NP {w['np']:.2f}, Auto {w['auto']:.2f}, TeleOp {w['tele']:.2f}, Pen {w['pen']:.2f}")
    return w

def score_dataframe(df_base: pd.DataFrame, w_perf: Dict[str, float], w_fit: Dict[str, float]) -> pd.DataFrame:
    """
    df_base must already have:
      np_norm, auto_norm, teleop_norm, pen_norm, base_matches
    Produces scout_score and alliance_fit_score as whole-number %.
    """
    df = df_base.copy()

    perf = (
        w_perf["np"] * df["np_norm"]
        + w_perf["auto"] * df["auto_norm"]
        + w_perf["tele"] * df["teleop_norm"]
        + w_perf["pen"] * df["pen_norm"]
    )
    scout = perf * confidence_factor(df["base_matches"])
    df["scout_score"] = (pd.to_numeric(scout, errors="coerce") * 100).round(0).astype("Int64")

    fit = (
        w_fit["auto"] * df["auto_norm"]
        + w_fit["tele"] * df["teleop_norm"]
        + w_fit["pen"] * df["pen_norm"]
    )
    df["alliance_fit_score"] = (pd.to_numeric(fit, errors="coerce") * 100).round(0).astype("Int64")
    return df

# ----------------------------
# FTCScout data fetch
# ----------------------------
TEAM_MATCHRECORDS_2025 = """
query MatchRecords($season: Int!, $skip: Int!, $take: Int!) {
  matchRecords(season: $season, skip: $skip, take: $take) {
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
def get_season_avgs_cached(season: int) -> Dict[int, Dict[str, Any]]:
    """
    Expensive season-wide scan (no region filter). Cached for 12 hours by Streamlit.
    Returns: {teamNumber: {season_matches, season_avg_np, season_avg_pen}}
    """
    agg: Dict[int, Dict[str, float]] = {}
    page_size = 300
    sleep_s = 0.02

    first = gql(TEAM_MATCHRECORDS_2025, {"season": season, "skip": 0, "take": page_size})
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

                d = agg.setdefault(int(t), {"np_sum": 0.0, "np_count": 0.0, "pen_sum": 0.0, "pen_count": 0.0})
                d["np_sum"] += float(np_val)
                d["np_count"] += 1.0
                if pen_val is not None:
                    d["pen_sum"] += float(pen_val)
                    d["pen_count"] += 1.0

    process_rows(mr["data"])
    skip = page_size
    while skip < total:
        page = gql(TEAM_MATCHRECORDS_2025, {"season": season, "skip": skip, "take": page_size})
        mr2 = page["matchRecords"]
        process_rows(mr2["data"])
        skip += page_size
        time.sleep(sleep_s)

    season_avgs: Dict[int, Dict[str, Any]] = {}
    for t, d in agg.items():
        np_count = int(d["np_count"])
        pen_count = int(d["pen_count"])
        season_avgs[int(t)] = {
            "season_matches": np_count,
            "season_avg_np": (d["np_sum"] / np_count) if np_count else None,
            "season_avg_pen": (d["pen_sum"] / pen_count) if pen_count else None,
        }
    return season_avgs

@st.cache_data(ttl=60 * 60 * 6, show_spinner=True)  # 6 hours
def fetch_event_roster(season: int, event_code: str) -> Tuple[Dict[str, Any], List[int]]:
    event = get_json(f"{REST_BASE}/events/{season}/{event_code}") or {}
    teps = get_json(f"{REST_BASE}/events/{season}/{event_code}/teams") or []
    roster = sorted({int(t["teamNumber"]) for t in teps if t.get("teamNumber") is not None})
    return event, roster

@st.cache_data(ttl=60, show_spinner=True)  # 60 seconds
def fetch_event_matches(season: int, event_code: str) -> List[Dict[str, Any]]:
    return get_json(f"{REST_BASE}/events/{season}/{event_code}/matches") or []

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_team_cached(team_number: int) -> Dict[str, Any]:
    return get_json(f"{REST_BASE}/teams/{team_number}") or {}

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def safe_quickstats(team_number: int, season: int, region: str) -> Dict[str, Any]:
    qs = get_json(f"{REST_BASE}/teams/{team_number}/quick-stats", params={"season": season, "region": region})
    return qs or {}

def _match_label(m: Dict[str, Any]) -> str:
    # We mostly care about qualifications ordering
    level = (m.get("tournamentLevel") or "").strip()
    mid = m.get("id")
    if level.lower().startswith("qual"):
        return f"Q{mid}"
    if level.lower().startswith("semi"):
        return f"SF{mid}"
    if level.lower().startswith("final"):
        return f"F{mid}"
    return f"M{mid}"

def compute_event_from_matches(matches: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], Set[int], bool, List[str], Dict[int, Dict[str, float]], Dict[int, List[float]]]:
    """
    From event matches (REST), compute:
      - event averages per team: NP, Pen, Auto, DC (TeleOp) + match count
      - active team set
      - any_played flag
      - match_labels ordered by match id
      - per-team per-match NP mapping (team -> {label: totalPointsNp})
      - per-team NP history list in match order (for momentum)
    """
    agg: Dict[int, Dict[str, float]] = {}
    active: Set[int] = set()
    any_played = False

    # sort by id (qual order). Some events may have id as matchId
    matches_sorted = sorted(matches, key=lambda x: (x.get("tournamentLevel") or "", x.get("series") or 0, x.get("id") or 0))
    match_labels: List[str] = []
    team_match_np: Dict[int, Dict[str, float]] = {}
    team_np_history: Dict[int, List[float]] = {}

    for m in matches_sorted:
        scores = m.get("scores")
        if not scores:
            continue
        if not m.get("hasBeenPlayed", True):
            continue

        any_played = True
        label = _match_label(m)
        match_labels.append(label)

        red = (scores.get("red") or {})
        blue = (scores.get("blue") or {})
        teams = m.get("teams") or []

        def add_for_alliance(alliance_name: str, alliance_scores: Dict[str, Any]):
            np_val = alliance_scores.get("totalPointsNp")
            pen_val = alliance_scores.get("penaltyPointsCommitted")
            auto_val = alliance_scores.get("autoPoints")
            dc_val = alliance_scores.get("dcPoints")

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

                d = agg.setdefault(t, {"np_sum": 0.0, "np_count": 0.0, "pen_sum": 0.0, "pen_count": 0.0, "auto_sum": 0.0, "dc_sum": 0.0})
                d["np_sum"] += float(np_val)
                d["np_count"] += 1.0
                if pen_val is not None:
                    d["pen_sum"] += float(pen_val)
                    d["pen_count"] += 1.0
                if auto_val is not None:
                    d["auto_sum"] += float(auto_val)
                if dc_val is not None:
                    d["dc_sum"] += float(dc_val)

                team_match_np.setdefault(t, {})[label] = float(np_val)
                team_np_history.setdefault(t, []).append(float(np_val))

        add_for_alliance("Red", red)
        add_for_alliance("Blue", blue)

    event_avgs: Dict[int, Dict[str, Any]] = {}
    for t, d in agg.items():
        np_count = int(d["np_count"])
        pen_count = int(d["pen_count"])
        event_avgs[t] = {
            "event_matches": np_count,
            "event_avg_np": (d["np_sum"] / np_count) if np_count else None,
            "event_avg_pen": (d["pen_sum"] / pen_count) if pen_count else None,
            "event_avg_auto": (d["auto_sum"] / np_count) if np_count else None,
            "event_avg_dc": (d["dc_sum"] / np_count) if np_count else None,
        }

    # Keep labels unique and stable (some events might have duplicates if id overlaps across levels)
    # If duplicates appear, we suffix them.
    seen = {}
    unique_labels = []
    for lab in match_labels:
        if lab not in seen:
            seen[lab] = 1
            unique_labels.append(lab)
        else:
            seen[lab] += 1
            unique_labels.append(f"{lab}_{seen[lab]}")
    # If we changed labels for uniqueness, remap team_match_np keys accordingly
    if unique_labels != match_labels:
        remap = dict(zip(match_labels, unique_labels))
        match_labels = unique_labels
        for t, mp in list(team_match_np.items()):
            team_match_np[t] = {remap.get(k, k): v for k, v in mp.items()}

    return event_avgs, active, any_played, match_labels, team_match_np, team_np_history

def build_dataframe(
    season: int,
    event_code: str,
    region: str,
    mode: str,
    include_inactive_override: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame, bool, int, int, List[str], Dict[int, Dict[str, float]], Dict[int, List[float]]]:
    # roster
    event, roster = fetch_event_roster(season, event_code)

    # matches -> event averages + match labels + per-team match NP
    matches = fetch_event_matches(season, event_code)
    event_avgs, active, any_played, match_labels, team_match_np, team_np_history = compute_event_from_matches(matches)

    # Mode logic for active filtering
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

    # season averages (cached)
    season_avgs = get_season_avgs_cached(season)

    rows = []
    for t in teams:
        qs = safe_quickstats(int(t), season, region)

        def stat(qs_obj, key):
            s = (qs_obj or {}).get(key) or {}
            return s.get("value"), s.get("rank")

        tot_v, tot_r = stat(qs, "tot")
        auto_v, _ = stat(qs, "auto")
        tele_v, _ = stat(qs, "dc")

        s = season_avgs.get(int(t), {})
        e = event_avgs.get(int(t), {})

        team_obj = fetch_team_cached(int(t))
        name = team_obj.get("name") or f"Team {t}"

        # momentum: last 3 NP matches at this event
        hist = team_np_history.get(int(t), [])
        mom3 = (sum(hist[-3:]) / len(hist[-3:])) if len(hist) > 0 else None

        base_auto = e.get("event_avg_auto") if mode == "Game Day (Live)" else auto_v
        base_tele = e.get("event_avg_dc") if mode == "Game Day (Live)" else tele_v

        rows.append({
            "team_number": int(t),
            "team_name": name,
            "active_today": (int(t) in active),

            "total_value": tot_v,
            "total_rank": tot_r,

            # On Game Day we override auto/tele to be event averages; pre-game uses quick-stats
            "auto_value": base_auto,
            "teleop_value": base_tele,

            "season_matches": s.get("season_matches", 0),
            "season_avg_np": s.get("season_avg_np"),
            "season_avg_pen": s.get("season_avg_pen"),

            "event_matches": e.get("event_matches", 0),
            "event_avg_np": e.get("event_avg_np"),
            "event_avg_pen": e.get("event_avg_pen"),

            "momentum_np_3": mom3,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return event, df, active_filtering_used, len(active), len(roster), match_labels, team_match_np, team_np_history

    # Choose baseline for scoring (season vs event)
    if mode == "Game Day (Live)":
        df["base_matches"] = pd.to_numeric(df["event_matches"], errors="coerce").fillna(0)
        df["base_np"] = pd.to_numeric(df["event_avg_np"], errors="coerce")
        df["base_pen"] = pd.to_numeric(df["event_avg_pen"], errors="coerce")
    else:
        df["base_matches"] = pd.to_numeric(df["season_matches"], errors="coerce").fillna(0)
        df["base_np"] = pd.to_numeric(df["season_avg_np"], errors="coerce")
        df["base_pen"] = pd.to_numeric(df["season_avg_pen"], errors="coerce")

    # normalize
    df["np_norm"] = minmax(df["base_np"], True)
    df["auto_norm"] = minmax(df["auto_value"], True)
    df["teleop_norm"] = minmax(df["teleop_value"], True)
    df["pen_norm"] = minmax(df["base_pen"], False)

    # momentum score (normalized 0-100)
    df["momentum_norm"] = minmax(df["momentum_np_3"], True)
    df["momentum_score"] = (pd.to_numeric(df["momentum_norm"], errors="coerce") * 100).round(0).astype("Int64")

    # attach per-match score columns (NP) for display (all modes; mainly shown on Game Day)
    for lab in match_labels:
        col = f"{lab}_NP"
        df[col] = df["team_number"].map(lambda tn: (team_match_np.get(int(tn), {}) or {}).get(lab))

    # round numeric display (raw columns)
    for c in [
        "total_value", "auto_value", "teleop_value",
        "season_avg_np", "season_avg_pen",
        "event_avg_np", "event_avg_pen",
        "momentum_np_3"
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(1)

    return event, df, active_filtering_used, len(active), len(roster), match_labels, team_match_np, team_np_history

# ----------------------------
# UI helpers
# ----------------------------
def apply_removed_teams(df: pd.DataFrame) -> pd.DataFrame:
    removed = st.session_state.get("removed_teams", set())
    if not removed:
        return df
    return df[~df["team_number"].isin(list(removed))].copy()

def pick_list_controls(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    removed = st.session_state.setdefault("removed_teams", set())

    st.markdown("### Pick List Controls")
    cols = st.columns([2, 1, 1])

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

def render_score_breakdown(df_scored: pd.DataFrame, w_perf: Dict[str, Any], w_fit: Dict[str, Any], key_prefix: str):
    with st.expander("ScoutScore breakdown (for one team)"):
        team = st.selectbox("Select team", df_scored["team_number"].tolist(), key=f"{key_prefix}_breakdown_team")
        r = df_scored[df_scored["team_number"] == team].iloc[0]
        st.write(
            {
                "Team": f"{int(team)} — {r['team_name']}",
                "Base matches (confidence)": int(r["base_matches"]),
                "NP (base)": r["base_np"],
                "Auto (value)": r["auto_value"],
                "TeleOp (value)": r["teleop_value"],
                "Penalty (base)": r["base_pen"],
                "Normalized components": {
                    "np_norm": float(r["np_norm"]) if pd.notna(r["np_norm"]) else None,
                    "auto_norm": float(r["auto_norm"]) if pd.notna(r["auto_norm"]) else None,
                    "teleop_norm": float(r["teleop_norm"]) if pd.notna(r["teleop_norm"]) else None,
                    "pen_norm": float(r["pen_norm"]) if pd.notna(r["pen_norm"]) else None,
                },
                "Weights": {
                    "ScoutScore": w_perf.get("raw", {}),
                    "AllianceFit": w_fit.get("raw", {}),
                },
                "Outputs": {
                    "ScoutScore": int(r["scout_score"]) if pd.notna(r["scout_score"]) else None,
                    "AllianceFit": int(r["alliance_fit_score"]) if pd.notna(r["alliance_fit_score"]) else None,
                }
            }
        )

def render_pick_list_tab(
    df_base: pd.DataFrame,
    tab_name: str,
    perf_defaults: Dict[str, int],
    fit_defaults: Dict[str, int],
    sort_mode: str,
    key_prefix: str,
    mode: str,
    match_labels: List[str],
):
    st.header(tab_name)

    if mode == "Game Day (Live)":
        st.info("Game Day ranking: ScoutScore & AllianceFit are computed from THIS EVENT's played matches (Auto/DC/NP/Pen).")

    w_perf = weights_ui(f"{key_prefix}_perf", "ScoutScore weights", perf_defaults, allow_pen_zero=True)
    st.divider()
    w_fit = weights_ui(f"{key_prefix}_fit", "AllianceFit weights", fit_defaults, allow_pen_zero=True)

    df_scored = score_dataframe(df_base, w_perf, w_fit)

    # Sort (pick lists can be model-based or FTCScout-based)
    if sort_mode == "FTCScout (total_rank)":
        df_scored = df_scored.sort_values(by=["total_rank", "total_value"], ascending=[True, False], na_position="last")
    else:
        df_scored = df_scored.sort_values(by=["scout_score", "alliance_fit_score", "total_value", "total_rank"], ascending=[False, False, False, True], na_position="last")

    df_scored = df_scored.reset_index(drop=True)
    df_scored = pick_list_controls(df_scored, key_prefix=key_prefix).reset_index(drop=True)
    df_scored.insert(0, "No.", range(1, len(df_scored) + 1))

    # Columns requested for Game Day and pick lists
    base_cols = ["No.", "team_number", "team_name", "scout_score", "alliance_fit_score", "total_value", "auto_value", "teleop_value", "season_avg_np"]
    if mode == "Game Day (Live)":
        match_cols = [f"{lab}_NP" for lab in match_labels]
        show_cols = base_cols + match_cols
    else:
        show_cols = base_cols + ["total_rank", "season_matches", "active_today"]

    st.dataframe(df_scored[show_cols], use_container_width=True, height=650)
    render_score_breakdown(df_scored, w_perf, w_fit, key_prefix=key_prefix)

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="FTC Scouting Dashboard", layout="wide")

st.title("FTC Scouting Dashboard")
st.caption("Data source: FTCScout API (not direct scraping of ftc-events.firstinspires.org).")

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

    sort_mode = st.selectbox("Pick list sort", ["Model (ScoutScore)", "FTCScout (total_rank)"], index=0)

    refresh = st.button("Refresh Now")

    st.divider()
    st.subheader("Pick List")
    if st.button("Reset removed teams"):
        st.session_state.removed_teams = set()
        st.rerun()

if refresh:
    st.session_state["force_event_refresh"] = time.time()
    st.rerun()

event, df_base, active_filtering_used, active_count, roster_count, match_labels, team_match_np, team_np_history = build_dataframe(
    season=int(season),
    event_code=event_code.strip(),
    region=region.strip(),
    mode=mode,
    include_inactive_override=include_inactive_override,
)

if df_base.empty:
    st.warning("No teams returned. Double-check the event code.")
    st.stop()

st.caption(
    f"Event: **{event.get('name','(unknown)')}** | Mode: **{mode}** | "
    f"Roster size: {roster_count} | Active today: {active_count} | Showing: {len(df_base)} | "
    f"Removed: {len(st.session_state.removed_teams)}"
)

if mode == "Game Day (Live)" and (not include_inactive_override) and (not active_filtering_used):
    st.info("No played matches found yet — showing full roster until matches begin.")

tabs = st.tabs(["Pre-Game (32240)", "Pick List A", "Pick List B", "Pick List C", "Momentum", "Compare Teams", "Team Detail"])

# ----------------------------
# Tab 0: Pre-Game (always rank by total_rank)
# ----------------------------
with tabs[0]:
    st.header(f"Pre-Game Analysis — Penguinauts {PENGUINAUTS_TEAM} vs this event roster")

    if mode == "Game Day (Live)":
        st.info("Game Day view: ScoutScore & AllianceFit are computed from THIS EVENT's played matches (Auto/DC/NP/Pen).")

    w_perf_default = {"np": 32, "auto": 22, "tele": 22, "pen": 2}
    w_fit_default = {"auto": 43, "tele": 25, "pen": 5}

    df_scored = score_dataframe(
        df_base,
        {"np": w_perf_default["np"]/100, "auto": w_perf_default["auto"]/100, "tele": w_perf_default["tele"]/100, "pen": w_perf_default["pen"]/100},
        {"auto": w_fit_default["auto"]/100, "tele": w_fit_default["tele"]/100, "pen": w_fit_default["pen"]/100},
    )

    # Always rank Pre-Game by FTCScout total_rank
    df_scored = df_scored.sort_values(by=["total_rank", "total_value"], ascending=[True, False], na_position="last").reset_index(drop=True)
    df_scored.insert(0, "No.", range(1, len(df_scored) + 1))

    if PENGUINAUTS_TEAM in df_scored["team_number"].values:
        idx = int(df_scored.index[df_scored["team_number"] == PENGUINAUTS_TEAM][0])
        row = df_scored.iloc[idx]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("32240 Rank (FTCScout)", f"{idx+1}/{len(df_scored)}")
        c2.metric("32240 ScoutScore", f"{row['scout_score']}%")
        c3.metric("32240 AllianceFit", f"{row['alliance_fit_score']}%")
        c4.metric("Total Value / Rank", f"{row['total_value']} / {row['total_rank']}")
    else:
        st.warning(f"Team {PENGUINAUTS_TEAM} not found in this roster.")

    st.subheader("All teams (full roster)")

    base_cols = ["No.", "team_number", "team_name", "scout_score", "alliance_fit_score", "total_value", "auto_value", "teleop_value", "season_avg_np"]
    if mode == "Game Day (Live)":
        match_cols = [f"{lab}_NP" for lab in match_labels]
        show_cols = base_cols + match_cols
    else:
        show_cols = base_cols + ["total_rank", "season_matches", "active_today"]

    st.dataframe(df_scored[show_cols], use_container_width=True, height=650)

# ----------------------------
# Pick List A/B/C
# ----------------------------
with tabs[1]:
    render_pick_list_tab(
        df_base=df_base,
        tab_name="Pick List A (Balanced)",
        perf_defaults={"np": 32, "auto": 22, "tele": 22, "pen": 2},
        fit_defaults={"auto": 43, "tele": 25, "pen": 5, "np": 0},
        sort_mode=sort_mode,
        key_prefix="pickA",
        mode=mode,
        match_labels=match_labels,
    )

with tabs[2]:
    render_pick_list_tab(
        df_base=df_base,
        tab_name="Pick List B (Low-pen emphasis)",
        perf_defaults={"np": 34, "auto": 20, "tele": 22, "pen": 4},
        fit_defaults={"auto": 38, "tele": 30, "pen": 7, "np": 0},
        sort_mode=sort_mode,
        key_prefix="pickB",
        mode=mode,
        match_labels=match_labels,
    )

with tabs[3]:
    render_pick_list_tab(
        df_base=df_base,
        tab_name="Pick List C (Auto-first)",
        perf_defaults={"np": 28, "auto": 30, "tele": 20, "pen": 2},
        fit_defaults={"auto": 50, "tele": 20, "pen": 10, "np": 0},
        sort_mode=sort_mode,
        key_prefix="pickC",
        mode=mode,
        match_labels=match_labels,
    )

# ----------------------------
# Momentum tab (last 3 matches NP)
# ----------------------------
with tabs[4]:
    st.header("Momentum (last 3 played matches)")
    if mode != "Game Day (Live)":
        st.info("Switch to Game Day to see event momentum.")
    else:
        # Show top momentum teams
        mom = df_base[["team_number", "team_name", "momentum_np_3", "momentum_score", "event_matches"]].copy()
        mom = mom.sort_values(by=["momentum_score", "event_matches"], ascending=[False, False], na_position="last")
        st.dataframe(mom, use_container_width=True, height=650)

# ----------------------------
# Compare Teams
# ----------------------------
with tabs[5]:
    st.header("Compare Teams")
    a, b = st.columns(2)

    team_list = df_base["team_number"].tolist()
    t1 = a.selectbox("Team 1", team_list, index=0)
    t2 = b.selectbox("Team 2", team_list, index=1 if len(team_list) > 1 else 0)

    r1 = df_base[df_base["team_number"] == t1].iloc[0]
    r2 = df_base[df_base["team_number"] == t2].iloc[0]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"{t1} — {r1['team_name']}")
        st.json({
            "Total Value": r1["total_value"],
            "Total Rank": r1["total_rank"],
            "Auto": r1["auto_value"],
            "TeleOp": r1["teleop_value"],
            "Season Avg NP": r1["season_avg_np"],
            "Event Avg NP": r1["event_avg_np"],
            "Event Matches": r1["event_matches"],
            "Momentum NP (last 3)": r1["momentum_np_3"],
            "Active Today": bool(r1["active_today"]),
        })
    with c2:
        st.subheader(f"{t2} — {r2['team_name']}")
        st.json({
            "Total Value": r2["total_value"],
            "Total Rank": r2["total_rank"],
            "Auto": r2["auto_value"],
            "TeleOp": r2["teleop_value"],
            "Season Avg NP": r2["season_avg_np"],
            "Event Avg NP": r2["event_avg_np"],
            "Event Matches": r2["event_matches"],
            "Momentum NP (last 3)": r2["momentum_np_3"],
            "Active Today": bool(r2["active_today"]),
        })

# ----------------------------
# Team Detail
# ----------------------------
with tabs[6]:
    st.header("Team Detail")
    t = st.selectbox("Select team", df_base["team_number"].tolist(), index=0)
    row = df_base[df_base["team_number"] == t].iloc[0]

    st.subheader(f"{t} — {row['team_name']}")
    st.write("**Quick overview**")
    st.write(
        f"- Total: {row['total_value']} (Rank {row['total_rank']})\n"
        f"- Auto: {row['auto_value']} | TeleOp: {row['teleop_value']}\n"
        f"- Season Avg NP: {row['season_avg_np']} | Season Matches: {row['season_matches']}\n"
        f"- Event Avg NP: {row['event_avg_np']} | Event Matches: {row['event_matches']}\n"
        f"- Momentum (last 3 NP): {row['momentum_np_3']} (Score {row['momentum_score']})\n"
        f"- Active today: {bool(row['active_today'])}"
    )

    if mode == "Game Day (Live)" and match_labels:
        st.subheader("Match-by-match NP")
        mp = team_match_np.get(int(t), {}) or {}
        # show as table in match order
        data = [{"Match": lab, "NP": mp.get(lab)} for lab in match_labels if lab in mp]
        st.dataframe(pd.DataFrame(data), use_container_width=True)

    st.button("Log out", on_click=st.logout)
    
st.markdown("---")
st.markdown("""
© 2026 Swara Tamirisa (32240). All rights reserved.  
Unauthorized use, duplication, or distribution of this scouting system,
data model, or scoring methodology is strictly prohibited.
""")
