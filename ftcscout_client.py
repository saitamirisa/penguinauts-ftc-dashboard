import time
import json, os
import requests

REST_BASE = "https://api.ftcscout.org/rest/v1"
GQL_URL = "https://api.ftcscout.org/graphql"

S = requests.Session()

MATCH_RECORDS_2025 = """
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
CACHE_DIR = "./ftc_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def disk_cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)

def disk_cache_get(name: str, ttl_seconds: int):
    p = disk_cache_path(name)
    if not os.path.exists(p):
        return None
    age = time.time() - os.path.getmtime(p)
    if age > ttl_seconds:
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def disk_cache_set(name: str, obj):
    p = disk_cache_path(name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _cache_file(name: str) -> str:
    return os.path.join(CACHE_DIR, name)

def cache_read_json(name: str, ttl_seconds: int):
    path = _cache_file(name)
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > ttl_seconds:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def cache_write_json(name: str, obj):
    path = _cache_file(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get_json(url: str, params=None, timeout=(15, 120)):
    r = S.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def gql(query: str, variables: dict, timeout=(15, 120)):
    r = S.post(GQL_URL, json={"query": query, "variables": variables}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if "errors" in payload:
        raise RuntimeError(json.dumps(payload["errors"], indent=2))
    return payload["data"]

def safe_quickstats(team_num: int, season: int, region: str):
    # FTCScout returns 404 if team has no events in that season/region
    try:
        return get_json(f"{REST_BASE}/teams/{team_num}/quick-stats", params={"season": season, "region": region})
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise

def fetch_event_roster(season: int, event_code: str):
    event = get_json(f"{REST_BASE}/events/{season}/{event_code}")
    teps = get_json(f"{REST_BASE}/events/{season}/{event_code}/teams")
    team_numbers = sorted({t["teamNumber"] for t in teps})
    return event, team_numbers

def fetch_team(team_num: int):
    return get_json(f"{REST_BASE}/teams/{team_num}")

def compute_event_np_penalties_and_active(season: int, event_code: str):
    """
    Uses event matches to:
      - determine which teams ACTUALLY played (active)
      - compute event-only avg NP and avg penalties
    """
    matches = get_json(f"{REST_BASE}/events/{season}/{event_code}/matches")

    agg = {}
    active = set()

    for m in matches:
        if not m.get("hasBeenPlayed"):
            continue

        scores = m.get("scores") or {}
        teams = m.get("teams") or []

        def process(alliance_name: str, score_obj: dict | None):
            if not score_obj:
                return
            np_val = score_obj.get("totalPointsNp")
            pen_val = score_obj.get("penaltyPointsCommitted")
            if np_val is None:
                return

            for p in teams:
                if p.get("alliance") != alliance_name:
                    continue
                if not p.get("onField"):
                    continue
                if p.get("noShow") or p.get("dq"):
                    continue

                t = p.get("teamNumber")
                if t is None:
                    continue

                active.add(t)
                d = agg.setdefault(t, {"np_sum": 0.0, "np_count": 0, "pen_sum": 0.0, "pen_count": 0})
                d["np_sum"] += float(np_val)
                d["np_count"] += 1
                if pen_val is not None:
                    d["pen_sum"] += float(pen_val)
                    d["pen_count"] += 1

        process("Red", scores.get("red"))
        process("Blue", scores.get("blue"))

    # Convert to averages
    event_avgs = {}
    for t, d in agg.items():
        event_avgs[t] = {
            "event_matches": d["np_count"],
            "event_avg_np": (d["np_sum"] / d["np_count"]) if d["np_count"] else None,
            "event_avg_pen": (d["pen_sum"] / d["pen_count"]) if d["pen_count"] else None,
        }

    return event_avgs, active

def compute_season_np_penalties_bulk(
    season: int,
    region: str,
    page_size: int = 250,
    sleep_s: float = 0.02,
    cache_ttl_seconds: int = 60 * 60 * 24 * 7,  # 7 days
    force_refresh: bool = False,
):
    """
    Expensive: scans matchRecords for the region/season to compute season averages.

    Disk cache:
      - Stored in ./ftc_cache/season_agg_{season}_{region}.json
      - Survives Streamlit restarts
    """
    cache_name = f"season_agg_{season}_{region}.json"

    if not force_refresh:
        cached = cache_read_json(cache_name, ttl_seconds=cache_ttl_seconds)
        if cached is not None:
            # keys were saved as strings; convert back to int
            return {int(k): v for k, v in cached.items()}

    agg = {}

    first = gql(MATCH_RECORDS_2025, {"season": season, "skip": 0, "take": page_size, "region": region})
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
        page = gql(MATCH_RECORDS_2025, {"season": season, "skip": skip, "take": page_size, "region": region})
        mr2 = page["matchRecords"]
        process_rows(mr2["data"])
        skip += page_size
        time.sleep(sleep_s)

    # averages
    season_avgs = {}
    for t, d in agg.items():
        season_avgs[t] = {
            "season_matches": d["np_count"],
            "season_avg_np": (d["np_sum"] / d["np_count"]) if d["np_count"] else None,
            "season_avg_pen": (d["pen_sum"] / d["pen_count"]) if d["pen_count"] else None,
        }

    # write cache (save keys as strings for JSON)
    cache_write_json(cache_name, {str(k): v for k, v in season_avgs.items()})

    return season_avgs

