#!/usr/bin/env python3
import math
import time

import pandas as pd
import altair as alt
import streamlit as st

from ftcscout_client import (
    fetch_event_roster,
    safe_quickstats,
    compute_event_np_penalties_and_active,
    compute_season_np_penalties_bulk,
)

st.set_page_config(page_title="FTC Scouting Dashboard", layout="wide")

PENGUINAUTS_TEAM = 32240


# ---------------------------
# Session defaults
# ---------------------------
def init_weights():
    if "scout_weights_ui" not in st.session_state:
        st.session_state.scout_weights_ui = {"np": 32, "auto": 22, "tele": 22, "eg": 22, "pen": 2}
    if "fit_weights_ui" not in st.session_state:
        st.session_state.fit_weights_ui = {"auto": 43, "eg": 27, "tele": 25, "pen": 5}
    if "removed_teams" not in st.session_state:
        st.session_state.removed_teams = set()
    st.session_state.setdefault("force_event_refresh", 0.0)


init_weights()


# ---------------------------
# Scoring helpers
# ---------------------------
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
    return 0.85 + 0.15 * (1.0 - (1.0 / (1.0 + n / 12.0)))


def normalize_weights(d: dict) -> dict:
    total = float(sum(d.values()))
    return {k: (float(v) / total if total else 0.0) for k, v in d.items()}


# ---------------------------
# Cached data access
# ---------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)  # 24 hours
def fetch_team_cached(team_num: int):
    from ftcscout_client import fetch_team
    return fetch_team(team_num)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=True)  # 24 hours
def get_season_avgs_cached(season: int, region: str, force_refresh: bool = False):
    # Use disk caching inside client (7d) + Streamlit caching (24h)
    return compute_season_np_penalties_bulk(
        season,
        region,
        page_size=300,
        cache_ttl_seconds=60 * 60 * 24 * 7,
        force_refresh=force_refresh,
    )


@st.cache_data(ttl=30, show_spinner=False)
def get_event_live_cached(season: int, event_code: str, refresh_token: float):
    # refresh_token is only to bust cache when user presses Refresh Now
    return compute_event_np_penalties_and_active(season, event_code)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def quickstats_cached(team_num: int, season: int, region: str):
    return safe_quickstats(team_num, season, region)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_event_roster_cached(season: int, event_code: str):
    return fetch_event_roster(season, event_code)


# ---------------------------
# Data build
# ---------------------------
def build_dataframe(
    season: int,
    event_code: str,
    region: str,
    mode: str,
    include_inactive_override: bool,
    rebuild_season: bool,
    scout_weights_ui: dict,
    fit_weights_ui: dict,
):
    # 1) roster
    event, roster = fetch_event_roster_cached(season, event_code)

    # 2) event-only NP/pen + active teams (only for Game Day)
    event_avgs, active = ({}, set())
    if mode == "Game Day (Live)":
        refresh_token = st.session_state.get("force_event_refresh", 0.0)
        event_avgs, active = get_event_live_cached(season, event_code, refresh_token)

    # Mode logic
    # - Pre-Game: always use roster (your request)
    # - Game Day: prefer active-only; if no played matches yet, fall back to roster
    roster_count = len(roster)
    active_count = len(active)

    if mode == "Pre-Game Analysis":
        teams = roster
        active_filtering_used = False
    else:
        if include_inactive_override:
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
    season_avgs = get_season_avgs_cached(season, region, force_refresh=rebuild_season)

    rows = []
    for t in teams:
        qs = quickstats_cached(t, season, region)

        def stat(qs_obj, key):
            s = (qs_obj or {}).get(key) or {}
            return s.get("value"), s.get("rank")

        tot_v, tot_r = stat(qs, "tot")
        auto_v, _ = stat(qs, "auto")
        tele_v, _ = stat(qs, "dc")
        eg_v, _ = stat(qs, "eg")

        s = season_avgs.get(t, {})
        e = event_avgs.get(t, {})

        team = fetch_team_cached(t) or {}

        rows.append(
            {
                "team_number": t,
                "team_name": team.get("name") or f"Team {t}",
                "active_today": (t in active) if mode == "Game Day (Live)" else None,
                "total_value": tot_v,
                "total_rank": tot_r,
                "auto_value": auto_v,
                "teleop_value": tele_v,
                "endgame_value": eg_v,
                "season_matches": s.get("season_matches"),
                "season_avg_np": s.get("season_avg_np"),
                "season_avg_pen": s.get("season_avg_pen"),
                "event_matches": e.get("event_matches"),
                "event_avg_np": e.get("event_avg_np"),
                "event_avg_pen": e.get("event_avg_pen"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return event, df, active_filtering_used, active_count, roster_count

    # Normalize + scores (baseline = season)
    df["np_norm"] = minmax(df["season_avg_np"], True)
    df["auto_norm"] = minmax(df["auto_value"], True)
    df["teleop_norm"] = minmax(df["teleop_value"], True)
    df["eg_norm"] = minmax(df["endgame_value"], True)
    df["pen_norm"] = minmax(df["season_avg_pen"], False)

    # Weights from UI (1–100), auto-normalize
    sw = normalize_weights(scout_weights_ui)
    fw = normalize_weights(fit_weights_ui)

    perf = (
        sw["np"] * df["np_norm"]
        + sw["auto"] * df["auto_norm"]
        + sw["tele"] * df["teleop_norm"]
        + sw["eg"] * df["eg_norm"]
        + sw["pen"] * df["pen_norm"]
    )
    df["scout_score"] = perf * confidence_factor(df["season_matches"])

    df["alliance_fit_score"] = (
        fw["auto"] * df["auto_norm"]
        + fw["eg"] * df["eg_norm"]
        + fw["tele"] * df["teleop_norm"]
        + fw["pen"] * df["pen_norm"]
    )

    # Present as whole-number %
    df["scout_score"] = (pd.to_numeric(df["scout_score"], errors="coerce") * 100).round(0).astype("Int64")
    df["alliance_fit_score"] = (pd.to_numeric(df["alliance_fit_score"], errors="coerce") * 100).round(0).astype("Int64")

    # Clean numeric display
    for c in [
        "total_value",
        "auto_value",
        "teleop_value",
        "endgame_value",
        "season_avg_np",
        "season_avg_pen",
        "event_avg_np",
        "event_avg_pen",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(1)

    # Default sorting: pick order (your latest preference)
    df = df.sort_values(
        by=["scout_score", "alliance_fit_score", "total_value", "total_rank"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    return event, df, active_filtering_used, active_count, roster_count


# ---------------------------
# UI
# ---------------------------
st.title("FTC Scouting Dashboard")

# ---------------------------
# Table rendering (keeps rank "No." in order when sorting)
# ---------------------------
def render_rank_table(df_in: pd.DataFrame, columns, height: int = 520, key: str = "rank_table"):
    """
    Shows a table with a dynamic rank column that ALWAYS stays 1..N after sorting.
    - If streamlit-aggrid is installed, rank updates automatically when the user sorts by clicking headers.
    - Otherwise, falls back to Streamlit's dataframe + server-side sort controls.
    """
    # --- Preferred: AgGrid (dynamic rowIndex-based rank) ---
    try:
        # NOTE: streamlit-aggrid has minor API differences across versions.
        # Import only the stable pieces to avoid falling back unexpectedly.
        from st_aggrid import AgGrid, GridOptionsBuilder  # type: ignore

        df_show = df_in.copy()

        # Build grid options
        gb = GridOptionsBuilder.from_dataframe(df_show[columns], enableRowGroup=False, enableValue=False, enablePivot=False)
        gb.configure_default_column(sortable=True, filter=True, resizable=True)

        # Insert a synthetic rank column that uses the displayed row index (updates with sorting/filtering)
        # Compute rank based on the displayed row order (updates after sort/filter).
        gb.configure_column(
            "No.",
            header_name="No.",
            valueGetter="node.rowIndex + 1",
            sortable=False,
            filter=False,
            width=80,
            pinned="left",
        )

        # Make sure the remaining columns exist (AgGrid needs them configured after No. sometimes)
        for c in columns:
            if c != "No.":
                gb.configure_column(c, header_name=c)

        go = gb.build()

        AgGrid(
            df_show[columns],
            gridOptions=go,
            height=height,
            # Use string form for maximum compatibility across streamlit-aggrid versions.
            update_mode="no_update",
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            key=key,
        )
        return
    except Exception:
        pass

    # --- Fallback: Streamlit dataframe ---
    # Streamlit's built-in dataframe sorting is client-side, so a "No." (rank) column cannot auto-renumber
    # when users click headers. In fallback mode we hide the No. column to avoid showing stale/blank values.
    st.info(
        "Tip: install **streamlit-aggrid** (`pip install streamlit-aggrid`) to keep the No. column renumbered "
        "when you sort by clicking headers."
    )
    df_show = df_in[columns].copy()
    if "No." in df_show.columns:
        df_show = df_show.drop(columns=["No."])
    st.dataframe(df_show, use_container_width=True, height=height, hide_index=True)

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

    refresh_event = st.button("Refresh Now (event only)")
    rebuild_season = st.button("Rebuild season cache (slow)")

    st.divider()
    st.subheader("Pick List Controls")
    if st.button("Reset removed teams"):
        st.session_state.removed_teams = set()

    if refresh_event:
        st.session_state["force_event_refresh"] = time.time()

    st.divider()
    st.subheader("Weight Tuning (1–100)")
    st.caption("Move sliders, then click **Apply weights**.")

    # Draft values (do not apply until user clicks)
    draft_scout = {
        "np": st.slider("Scout: NP", 1, 100, int(st.session_state.scout_weights_ui["np"]), 1),
        "auto": st.slider("Scout: Auto", 1, 100, int(st.session_state.scout_weights_ui["auto"]), 1),
        "tele": st.slider("Scout: TeleOp", 1, 100, int(st.session_state.scout_weights_ui["tele"]), 1),
        "eg": st.slider("Scout: Endgame", 1, 100, int(st.session_state.scout_weights_ui["eg"]), 1),
        "pen": st.slider("Scout: Penalties", 1, 100, int(st.session_state.scout_weights_ui["pen"]), 1),
    }

    st.divider()

    draft_fit = {
        "auto": st.slider("Fit: Auto", 1, 100, int(st.session_state.fit_weights_ui["auto"]), 1),
        "eg": st.slider("Fit: Endgame", 1, 100, int(st.session_state.fit_weights_ui["eg"]), 1),
        "tele": st.slider("Fit: TeleOp", 1, 100, int(st.session_state.fit_weights_ui["tele"]), 1),
        "pen": st.slider("Fit: Penalties", 1, 100, int(st.session_state.fit_weights_ui["pen"]), 1),
    }

    cA, cB = st.columns(2)
    apply_weights = cA.button("✅ Apply weights")
    reset_weights = cB.button("Reset")

    if reset_weights:
        st.session_state.scout_weights_ui = {"np": 32, "auto": 22, "tele": 22, "eg": 22, "pen": 2}
        st.session_state.fit_weights_ui = {"auto": 43, "eg": 27, "tele": 25, "pen": 5}
        st.rerun()

    if apply_weights:
        st.session_state.scout_weights_ui = draft_scout
        st.session_state.fit_weights_ui = draft_fit
        st.success("Weights applied.")
        st.rerun()

# Build data (must happen BEFORE any df usage)
event, df, active_filtering_used, active_count, roster_count = build_dataframe(
    season=season,
    event_code=event_code.strip(),
    region=region.strip(),
    mode=mode,
    include_inactive_override=include_inactive_override,
    rebuild_season=rebuild_season,
    scout_weights_ui=st.session_state.scout_weights_ui,
    fit_weights_ui=st.session_state.fit_weights_ui,
)

if df.empty:
    st.warning(
        "No teams returned. Double-check the event code. "
        "If this is Game Day and matches haven't started yet, switch to Pre-Game Analysis."
    )
    st.stop()

st.caption(
    f"Event: **{event.get('name')}** | Mode: **{mode}** | "
    f"Roster size: {roster_count} | Active today: {active_count} | Showing: {len(df)} | "
    f"Removed: {len(st.session_state.removed_teams)}"
)

if mode == "Game Day (Live)" and (not include_inactive_override) and (not active_filtering_used):
    st.info("No played matches found yet — showing full roster until matches begin.")

tabs = st.tabs(["Pre-Game (32240)", "Pick List", "Compare Teams", "Team Detail"])

# ---------------------------
# Pre-Game tab
# ---------------------------
with tabs[0]:
    st.subheader("Pre-Game Analysis — Penguinauts 32240 vs this event roster")

    show_all = st.checkbox("Show full roster table (all teams)", value=True)

    df_ranked = df.copy()
    # "No." is computed dynamically in the table renderer (AgGrid). Keep it blank in fallback mode.
    df_ranked["No."] = ""

    if PENGUINAUTS_TEAM in df_ranked["team_number"].values:
        row = df_ranked[df_ranked["team_number"] == PENGUINAUTS_TEAM].iloc[0]
        rank_no = int(df_ranked.index[df_ranked["team_number"] == PENGUINAUTS_TEAM][0] + 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("32240 Rank (by ScoutScore)", f"{rank_no}/{len(df_ranked)}")
        c2.metric("32240 ScoutScore", f"{row['scout_score']}%")
        c3.metric("32240 AllianceFit", f"{row['alliance_fit_score']}%")
        c4.metric("Total Value / Rank", f"{row['total_value']} / {row['total_rank']}")

        if show_all:
            render_rank_table(
                df_ranked,
                columns=[
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
                    ],
                height=520,
                key="pregame_all"
            )
        else:
            start = max(rank_no - 6, 0)
            end = min(rank_no + 5, len(df_ranked))
            around = df_ranked.iloc[start:end].copy()
            st.write("Teams around 32240 in the ranking:")
            render_rank_table(
                around,
                columns=[
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
                    ],
                height=320,
                key="pregame_around"
            )

        # 32240 vs field average chart
        metrics = ["auto_value", "teleop_value", "endgame_value", "season_avg_np", "season_avg_pen"]
        field_avg = df[metrics].mean(numeric_only=True)

        chart_df = pd.DataFrame(
            {
                "Metric": ["Auto", "TeleOp", "Endgame", "Season NP", "Season Pen"],
                "32240": [
                    row["auto_value"],
                    row["teleop_value"],
                    row["endgame_value"],
                    row["season_avg_np"],
                    row["season_avg_pen"],
                ],
                "Field Avg": [
                    field_avg["auto_value"],
                    field_avg["teleop_value"],
                    field_avg["endgame_value"],
                    field_avg["season_avg_np"],
                    field_avg["season_avg_pen"],
                ],
            }
        ).melt(id_vars=["Metric"], var_name="Team", value_name="Value")

        chart_df["Value"] = pd.to_numeric(chart_df["Value"], errors="coerce")

        st.altair_chart(
            alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Metric:N", sort=None),
                y=alt.Y("Value:Q"),
                color="Team:N",
            ).properties(height=300),
            use_container_width=True,
        )
    else:
        st.warning("Team 32240 is not in this event roster. Choose a different event code.")

# ---------------------------
# Pick List tab
# ---------------------------
with tabs[1]:
    st.subheader("Pick List")

    df_pick = df[~df["team_number"].isin(st.session_state.removed_teams)].copy()

    colA, colB = st.columns([2, 1])

    with colA:
        st.write("Sorted by: scout_score → alliance_fit_score → total_value → total_rank")
        show_cols = [
            "team_number",
            "team_name",
            "active_today",
            "scout_score",
            "alliance_fit_score",
            "total_value",
            "total_rank",
            "auto_value",
            "teleop_value",
            "endgame_value",
            "season_avg_np",
            "season_avg_pen",
            "event_avg_np",
            "event_avg_pen",
        ]
        st.dataframe(df_pick[show_cols], use_container_width=True, height=560, hide_index=True)

    with colB:
        st.markdown("### Remove a team (picked / not available)")
        to_remove = st.selectbox(
            "Select team",
            options=df_pick["team_number"].tolist(),
            format_func=lambda x: f"{x}",
        )
        if st.button("Remove selected team"):
            st.session_state.removed_teams.add(int(to_remove))
            st.success(f"Removed team {to_remove}")
            st.rerun()

        st.markdown("### Removed teams")
        removed_sorted = sorted(list(st.session_state.removed_teams))
        st.write(removed_sorted if removed_sorted else "None")

# ---------------------------
# Compare tab
# ---------------------------
with tabs[2]:
    st.subheader("Compare 2 Teams")

    team_nums = df["team_number"].tolist()
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        t1 = st.selectbox("Team A", options=team_nums, index=0)
    with c2:
        t2 = st.selectbox("Team B", options=team_nums, index=1 if len(team_nums) > 1 else 0)

    a = df[df["team_number"] == t1].iloc[0]
    b = df[df["team_number"] == t2].iloc[0]

    compare_metrics = [
        ("Scout Score (%)", "scout_score"),
        ("Alliance Fit (%)", "alliance_fit_score"),
        ("Total Value", "total_value"),
        ("Total Rank", "total_rank"),
        ("Auto", "auto_value"),
        ("TeleOp", "teleop_value"),
        ("Endgame", "endgame_value"),
        ("Season Avg NP", "season_avg_np"),
        ("Season Avg Pen", "season_avg_pen"),
        ("Event Avg NP", "event_avg_np"),
        ("Event Avg Pen", "event_avg_pen"),
    ]

    comp_df = pd.DataFrame(
        {
            "Metric": [m[0] for m in compare_metrics],
            "Team A": [a[m[1]] for m in compare_metrics],
            "Team B": [b[m[1]] for m in compare_metrics],
        }
    )

    with c3:
        st.dataframe(comp_df, use_container_width=True, height=420, hide_index=True)

    chart_df = comp_df[
        comp_df["Metric"].isin(["Auto", "TeleOp", "Endgame", "Season Avg NP", "Season Avg Pen"])
    ].melt(id_vars=["Metric"], var_name="Team", value_name="Value")
    chart_df["Value"] = pd.to_numeric(chart_df["Value"], errors="coerce")

    st.altair_chart(
        alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Metric:N", sort=None),
            y=alt.Y("Value:Q"),
            color="Team:N",
            column="Team:N",
        ).properties(height=260),
        use_container_width=True,
    )

# ---------------------------
# Team Detail tab
# ---------------------------
with tabs[3]:
    st.subheader("Team Detail")

    t = st.selectbox("Select a team", options=df["team_number"].tolist())
    row = df[df["team_number"] == t].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scout Score", f"{row['scout_score']}%")
    k2.metric("Alliance Fit", f"{row['alliance_fit_score']}%")
    k3.metric("Total Value", f"{row['total_value']}")
    k4.metric("Total Rank", f"{row['total_rank']}")

    detail = pd.DataFrame(
        {
            "Component": [
                "NP (Season)",
                "Penalties (Season)",
                "NP (Event)",
                "Penalties (Event)",
                "Auto",
                "TeleOp",
                "Endgame",
            ],
            "Value": [
                row["season_avg_np"],
                row["season_avg_pen"],
                row["event_avg_np"],
                row["event_avg_pen"],
                row["auto_value"],
                row["teleop_value"],
                row["endgame_value"],
            ],
        }
    )
    detail["Value"] = pd.to_numeric(detail["Value"], errors="coerce")

    st.altair_chart(
        alt.Chart(detail).mark_bar().encode(
            x=alt.X("Component:N", sort=None),
            y=alt.Y("Value:Q"),
        ).properties(height=320),
        use_container_width=True,
    )

    with st.expander("Raw fields"):
        st.json(row.to_dict())
