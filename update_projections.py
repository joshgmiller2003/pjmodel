#!/usr/bin/env python3
"""NBA PJ Model - daily pipeline.

Each run:
  1. Ingests finished games (box scores) from ESPN into data/player_log.json
  2. Settles yesterday's predictions: updates residuals, record and settled bets
  3. Projects today's/tomorrow's games and player props
  4. Writes projections_data.json, which index.html loads

Learning loop: every prediction is frozen before tip-off in data/pending.json.
When the game ends, (actual - model) becomes a residual that nudges future
projections for that player/stat, with a weight (alpha) that fades over time.

Env vars: ODDS_API_KEY (optional, enables player props from DraftKings via
the-odds-api.com), MAX_PROP_EVENTS (default 30, i.e. every game).
"""
import json, math, os, re, sys, time, unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "projections_data.json"
LOG_PATH = ROOT / "data" / "player_log.json"
PENDING_PATH = ROOT / "data" / "pending.json"
POTD_LOG = ROOT / "data" / "potd_log.json"      # permanent record of every Pick of the Day
LADDER_PATH = ROOT / "data" / "ladder.json"     # ladder challenge: every step, every attempt
RECORDS_DIR = ROOT / "data" / "records"         # permanent bet ledger: one file per month + a summary
LADDER_START = float(os.environ.get("LADDER_START", "10") or 10)
LADDER_GOAL = float(os.environ.get("LADDER_GOAL", "1000") or 1000)
ALT_MARKETS = {"player_points_alternate": "PTS", "player_rebounds_alternate": "REB",
               "player_assists_alternate": "AST", "player_threes_alternate": "3PM"}
ALT_GAMES = int(os.environ.get("ALT_LADDER_GAMES", "3") or 3)   # games per day that get alt-line ladders

SEASON = "2026-27"
SEASON_START = date(2026, 10, 20)
ET = ZoneInfo("America/New_York")
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ODDS = "https://api.the-odds-api.com/v4/sports/basketball_nba"
ODDS_KEY = os.environ.get("ODDS_API_KEY", "").strip()
MAX_PROP_EVENTS = int(os.environ.get("MAX_PROP_EVENTS", "30"))
DAILY_CREDIT_LIMIT = int(os.environ.get("DAILY_CREDIT_LIMIT", "150") or 150)   # hard cap per ET day
MONTHLY_CREDIT_LIMIT = int(os.environ.get("MONTHLY_CREDIT_LIMIT", "5000") or 5000)  # this site's share of the plan
ODDS_RESET_DAY = int(os.environ.get("ODDS_RESET_DAY", "1") or 1)               # day of month the plan resets
CREDITS_PATH = Path(__file__).resolve().parent / "data" / "credits.json"

PROP_MARKETS = {
    "player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST",
    "player_threes": "3PM",
    "player_points_rebounds_assists": "PRA", "player_points_rebounds": "PR",
    "player_points_assists": "PA", "player_rebounds_assists": "RA",
}
ABBR_FIX = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "WSH": "WAS", "UTAH": "UTA"}
NAME_FIX = {"Los Angeles Clippers": "LA Clippers"}
TEAM_NAMES = {"ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers", "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"}

HFA = 2.3            # home-court advantage, points
SHRINK_K = 8         # pseudo-games of league-average shrinkage for team ratings
GAME_SD = 12.0       # NBA single-game margin standard deviation
RES_BETA = 0.30      # how fast residuals adapt
BET_EDGE_GAME = 1.0  # min edge to count a game bet
BET_EDGE_PROP = 0.8  # min edge to count a prop bet
WIN_UNITS = 0.91     # +1 unit risked at -110


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------- helpers
S = requests.Session()
S.headers["User-Agent"] = "pj-model/1.0 (github pages project)"


def get(url, **params):
    for i in range(3):
        try:
            r = S.get(url, params=params or None, timeout=25)
            if r.status_code == 200:
                return r.json()
            log(f"HTTP {r.status_code} for {url}")
            if r.status_code in (401, 402, 403, 404, 422, 429):
                return None
        except (requests.RequestException, ValueError) as e:
            log(f"request error {url}: {e}")
        time.sleep(2 * (i + 1))
    return None


class Credits:
    """Hard daily budget for The Odds API. Every paid call goes through spend()."""

    def __init__(self, now):
        self.now = now
        self.day = now.date().isoformat()
        st = load(CREDITS_PATH, {})
        self.state = st if st.get("date") == self.day else {"date": self.day, "used_today": 0, "calls": [],
                                                              "remaining": st.get("remaining"), "history": st.get("history", []),
                                                              "adjust": st.get("adjust")}
        self.state["limit"] = DAILY_CREDIT_LIMIT
        self.skipped = []

    def days_to_reset(self):
        d = self.now.date()
        nxt = d.replace(day=min(ODDS_RESET_DAY, 28))
        if nxt <= d:
            nxt = (nxt.replace(day=1) + timedelta(days=32)).replace(day=min(ODDS_RESET_DAY, 28))
        return max(1, (nxt - d).days)

    def cycle_start(self):
        d = self.now.date()
        start = d.replace(day=min(ODDS_RESET_DAY, 28))
        if start > d:
            start = (start.replace(day=1) - timedelta(days=1)).replace(day=min(ODDS_RESET_DAY, 28))
        return start.isoformat()

    def month_used(self):
        """Credits THIS site has used since the plan last reset (the key may be shared with other sites)."""
        cs = self.cycle_start()
        adj = self.state.get("adjust") or {}
        extra = adj.get("used", 0) if adj.get("cycle") == cs else 0       # credits used before tracking existed
        return sum(h["used"] for h in self.state.get("history", []) if cs <= h["date"] < self.day) + self.state["used_today"] + extra

    def allowance(self):
        """What is still allowed today: the daily cap, and never more than an even share of what is left in the plan."""
        left_today = DAILY_CREDIT_LIMIT - self.state["used_today"]
        # this site's monthly budget, spread evenly over the days left in the cycle
        month_left = MONTHLY_CREDIT_LIMIT - self.month_used()
        site_share = (month_left + self.state["used_today"]) // self.days_to_reset() - self.state["used_today"]
        left_today = min(left_today, site_share, month_left)
        rem = self.state.get("remaining")
        if rem is not None:
            # even share of the plan: what was left at the start of today, split over the days until it resets
            share = (int(rem) + self.state["used_today"]) // self.days_to_reset() - self.state["used_today"]
            left_today = min(left_today, share, int(rem))
        return max(0, left_today)

    def spend(self, url, est, label, **params):
        if not ODDS_KEY:
            return None
        if est > self.allowance():
            self.skipped.append(label)
            log(f"credits: skipped {label} (needs ~{est}, {self.allowance()} left today)")
            return None
        try:
            r = S.get(url, params={**params, "apiKey": ODDS_KEY}, timeout=25)
        except requests.RequestException as e:
            log(f"odds request error {label}: {e}")
            return None
        cost = int(num(r.headers.get("x-requests-last", est)))
        rem = r.headers.get("x-requests-remaining")
        self.state["used_today"] += cost
        if rem is not None:
            self.state["remaining"] = int(num(rem))
        self.state["calls"].append({"time": datetime.now(ET).strftime("%I:%M %p"), "call": label, "cost": cost})
        if r.status_code != 200:
            log(f"odds HTTP {r.status_code} for {label}")
            return None
        try:
            return r.json()
        except ValueError:
            return None

    def save(self):
        st = self.state
        st["calls"] = st["calls"][-80:]
        hist = [h for h in st.get("history", []) if h["date"] != self.day]
        hist.insert(0, {"date": self.day, "used": st["used_today"], "limit": DAILY_CREDIT_LIMIT})
        st["history"] = hist[:60]
        st["days_to_reset"] = self.days_to_reset()
        st["month_used"] = self.month_used()
        st["month_limit"] = MONTHLY_CREDIT_LIMIT
        save(CREDITS_PATH, st, indent=1)


CREDITS = None


def load(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save(path, obj, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, separators=None if indent else (",", ":")), encoding="utf-8")
    tmp.replace(path)


def abbr(a):
    a = (a or "").upper()
    return ABBR_FIX.get(a, a)


def norm(name):
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def made(v):  # "3-7" -> 3
    try:
        return float(str(v).split("-")[0])
    except ValueError:
        return 0.0


def att(v):  # "3-7" -> 7
    try:
        return float(str(v).split("-")[1])
    except (ValueError, IndexError):
        return 0.0


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def rnd(x, n=1):
    return round(x + 0.0, n)


def sgn(x):
    return f"+{x:g}" if x > 0 else f"{x:g}"


# --------------------------------------------------------------- ESPN data
def scoreboard(d):
    j = get(f"{ESPN}/scoreboard", dates=d.strftime("%Y%m%d"), limit=50)
    return (j or {}).get("events") if j is not None else None


def parse_lines(comp, home):
    o = (comp.get("odds") or [None])[0]
    if not o:
        return None
    details = (o.get("details") or "").strip()
    spread = None
    if details.upper() in ("EVEN", "PK", "PICK"):
        spread = 0.0
    else:
        m = re.match(r"([A-Za-z]{2,4})\s+([+-]?\d+(?:\.\d+)?)", details)
        if m:
            val = float(m.group(2))
            spread = val if abbr(m.group(1)) == home else -val
    total = o.get("overUnder")
    if spread is None or total is None:
        return None

    def ml(side):
        v = ((o.get(side) or {}).get("moneyLine"))
        if v is None:   # newer ESPN shape: odds.moneyline.home.close.odds
            mk = (o.get("moneyline") or {}).get("home" if side == "homeTeamOdds" else "away") or {}
            v = (mk.get("close") or {}).get("odds") or (mk.get("open") or {}).get("odds")
        if v is None:
            return "N/A"
        v = str(v)
        return v if v.startswith(("+", "-")) or v == "EVEN" else (f"+{v}" if num(v) > 0 else v)
    return {"spread": sgn(spread), "total": float(total),
            "ml_away": ml("awayTeamOdds"), "ml_home": ml("homeTeamOdds"),
            "provider": (o.get("provider") or {}).get("name", "")}


def parse_event(ev):
    comp = ev["competitions"][0]
    t = {c["homeAway"]: c for c in comp["competitors"]}
    away, home = abbr(t["away"]["team"]["abbreviation"]), abbr(t["home"]["team"]["abbreviation"])
    state = ev["status"]["type"].get("state")
    status = {"post": "Final", "in": "Live"}.get(state, "Upcoming")
    dt = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone(ET)
    g = {"id": f"{away}@{home}", "event_id": str(ev["id"]), "date": dt.date().isoformat(),
         "time": dt.strftime("%I:%M %p ET").lstrip("0"), "away": away, "home": home,
         "status": status, "names": {away: t["away"]["team"].get("displayName", ""),
                                     home: t["home"]["team"].get("displayName", "")}}
    if status != "Upcoming":
        g["score"] = {"away": int(num(t["away"].get("score"))), "home": int(num(t["home"].get("score")))}
    g["_lines"] = parse_lines(comp, home)
    return g


def parse_box(summary):
    rows = []
    for tm in (summary.get("boxscore") or {}).get("players", []):
        team = abbr(tm["team"].get("abbreviation"))
        for blk in tm.get("statistics", []):
            labels = blk.get("labels") or blk.get("names") or []
            for a in blk.get("athletes", []):
                vals = a.get("stats") or []
                if a.get("didNotPlay") or len(vals) != len(labels):
                    continue
                r = dict(zip(labels, vals))
                mins = num(r.get("MIN"))
                if mins <= 0:
                    continue
                rows.append({"n": a["athlete"]["displayName"], "t": team, "min": mins,
                             "pos": ((a["athlete"].get("position") or {}).get("abbreviation") or "")[:1],
                             "pts": num(r.get("PTS")), "reb": num(r.get("REB")), "ast": num(r.get("AST")),
                             "tpm": made(r.get("3PT")), "stl": num(r.get("STL")), "blk": num(r.get("BLK")),
                             "fgm": made(r.get("FG")), "fga": att(r.get("FG")), "tpa": att(r.get("3PT")),
                             "ftm": made(r.get("FT")), "fta": att(r.get("FT")), "to": num(r.get("TO")),
                             "oreb": num(r.get("OREB")), "dreb": num(r.get("DREB")), "pf": num(r.get("PF")),
                             "pm": num(str(r.get("+/-", 0)).replace("+", ""))})
    return rows


def ingest(plog, today):
    """Pull any finished games we have not logged yet."""
    start = SEASON_START
    if plog.get("through"):
        start = max(SEASON_START, date.fromisoformat(plog["through"]) - timedelta(days=2))
    d, new = start, 0
    while d <= today:
        evs = scoreboard(d)
        if evs is None:
            log(f"scoreboard failed for {d}; will retry next run")
            break
        for ev in evs:
            g = parse_event(ev)
            if g["status"] != "Final" or g["event_id"] in plog["games"]:
                continue
            box = get(f"{ESPN}/summary", event=g["event_id"])
            players = parse_box(box) if box else []
            if not players:
                continue
            plog["games"][g["event_id"]] = {
                "date": g["date"], "away": g["away"], "home": g["home"],
                "away_pts": g["score"]["away"], "home_pts": g["score"]["home"], "players": players}
            new += 1
        plog["through"] = d.isoformat()
        d += timedelta(days=1)
    log(f"ingested {new} new finished games; log holds {len(plog['games'])}")


# ------------------------------------------------------------------- model
def team_ratings(plog):
    """Shrunk offense/defense per team from finished games."""
    rows = {}
    allpts = []
    for g in sorted(plog["games"].values(), key=lambda x: x["date"]):
        rows.setdefault(g["home"], []).append((g["home_pts"], g["away_pts"]))
        rows.setdefault(g["away"], []).append((g["away_pts"], g["home_pts"]))
        allpts += [g["home_pts"], g["away_pts"]]
    lg = sum(allpts) / len(allpts) if allpts else 114.0
    out = {}
    for t, r in rows.items():
        n = len(r)
        out[t] = {"off": (sum(a for a, _ in r) + SHRINK_K * lg) / (n + SHRINK_K),
                  "def": (sum(b for _, b in r) + SHRINK_K * lg) / (n + SHRINK_K), "n": n}
    return out, lg


def model_game(g, ratings, lg):
    h = ratings.get(g["home"], {"off": lg, "def": lg})
    a = ratings.get(g["away"], {"off": lg, "def": lg})
    home_pts = (h["off"] + a["def"]) / 2 + HFA / 2
    away_pts = (a["off"] + h["def"]) / 2 - HFA / 2
    margin = home_pts - away_pts                      # home perspective
    p_home = phi(margin / GAME_SD)
    return {"spread": rnd(-margin), "total": rnd(home_pts + away_pts),
            "ml_home": rnd(p_home, 2), "ml_away": rnd(1 - p_home, 2)}


def player_index(plog):
    idx = {}
    for g in sorted(plog["games"].values(), key=lambda x: x["date"]):
        for p in g["players"]:
            home = p["t"] == g["home"]
            idx.setdefault(norm(p["n"]), []).append({**p, "date": g["date"], "home": home,
                                                     "opp": g["away"] if home else g["home"]})
    return idx


def wmean(vals, decay):
    w = [decay ** i for i in range(len(vals))]
    return sum(v * x for v, x in zip(vals, w)) / sum(w)


DEF_STATS = ("pts", "reb", "ast", "tpm", "stl", "blk")
DEF_CAP = 0.10       # opponent defense can move a stat by at most +/-10%
PACE_CAP = 0.05      # expected game pace can move a stat by at most +/-5%
MATCHUP_CAP = 0.12   # combined matchup cap
STAT_KEY = {"PTS": "pts", "REB": "reb", "AST": "ast", "3PM": "tpm", "BLK": "blk", "STL": "stl"}


def _poss(rows):
    return sum(r.get("fga", 0) - r.get("oreb", 0) + r.get("to", 0) + 0.44 * r.get("fta", 0) for r in rows)


def _clamp(x, cap):
    return min(1 + cap, max(1 - cap, x))


POS_PATH = ROOT / "data" / "positions.json"
ESPN_TEAM = {"GSW": "gs", "NOP": "no", "NYK": "ny", "SAS": "sa", "UTA": "utah", "WAS": "wsh"}
USAGE_KEEP = 0.5     # share of a missing teammate's leftover production that the rest of the roster picks up
INJ_CAP = 0.20       # injury boost can raise a stat by at most +20%


def refresh_positions(today):
    """G/F/C for every player from ESPN team rosters (free, no Odds API). Once a day; keeps the old map on failure."""
    cur = load(POS_PATH, {"pos": {}})
    if cur.get("updated") == today.isoformat() and cur.get("pos"):
        return cur["pos"]
    pos, ok = dict(cur.get("pos", {})), 0
    for t in TEAM_NAMES:
        j = get(f"{ESPN}/teams/{ESPN_TEAM.get(t, t.lower())}/roster")
        if not j:
            continue
        ok += 1
        for a in j.get("athletes", []):
            p = ((a.get("position") or {}).get("abbreviation") or "")[:1]
            if p in ("G", "F", "C"):
                pos[a["displayName"]] = p
    if ok:
        save(POS_PATH, {"note": "Player positions (G/F/C) from ESPN team rosters. Refreshed automatically each day.",
                        "updated": today.isoformat(), "pos": dict(sorted(pos.items()))}, indent=0)
    log(f"positions: {len(pos)} players ({ok}/30 rosters refreshed)")
    return pos


def team_defense(plog, pos=None):
    """What each team allows in every prop stat, from ESPN box scores already saved in data/player_log.json
    (no API calls). Team level: per 100 possessions. By position (G/F/C): per minute played by opposing
    players at that position, with the team's pace taken out so it is not counted twice. Everything is shrunk
    toward league average (SHRINK_K pseudo-games) so a few odd games cannot swing it."""
    P = {norm(k): v for k, v in (pos or {}).items()}
    agg, lg, lg_poss, n = {}, {k: 0.0 for k in DEF_STATS}, 0.0, 0
    lgp = {q: {"min": 0.0, **{k: 0.0 for k in DEF_STATS}} for q in "GFC"}
    for g in plog["games"].values():
        h, a = g["home"], g["away"]
        by = {t: [p for p in g["players"] if p["t"] == t] for t in (h, a)}
        if not by[h] or not by[a]:
            continue
        gp = (_poss(by[h]) + _poss(by[a])) / 2
        if gp <= 50:
            continue
        for t, o in ((h, a), (a, h)):
            r = agg.setdefault(t, {"n": 0, "poss": 0.0, **{k: 0.0 for k in DEF_STATS},
                                   "bypos": {q: {"min": 0.0, **{k: 0.0 for k in DEF_STATS}} for q in "GFC"}})
            r["n"] += 1
            r["poss"] += gp
            for k in DEF_STATS:
                v = sum(p.get(k, 0) for p in by[o])
                r[k] += v
                lg[k] += v
            for p in by[o]:
                q = P.get(norm(p["n"]))
                if not q:
                    continue
                for tgt in (r["bypos"][q], lgp[q]):
                    tgt["min"] += p["min"]
                    for k in DEF_STATS:
                        tgt[k] += p.get(k, 0)
        lg_poss += 2 * gp
        n += 2
    if not n:
        return {"teams": {}, "pace": 100.0}
    lg_pace = lg_poss / n
    rate = {k: lg[k] / lg_poss for k in DEF_STATS}            # league stat per possession
    out = {}
    for t, r in agg.items():
        poss = r["poss"] + SHRINK_K * lg_pace
        pace = (r["poss"] + SHRINK_K * lg_pace) / (r["n"] + SHRINK_K)
        row = {"n": r["n"], "pace": pace,
               **{k: ((r[k] + SHRINK_K * lg_pace * rate[k]) / poss) / rate[k] if rate[k] else 1.0 for k in DEF_STATS}}
        row["pos"] = {}
        for q in "GFC":
            L, R = lgp[q], r["bypos"][q]
            if not L["min"]:
                continue
            shr = SHRINK_K * L["min"] / n                        # pseudo-minutes: K games of a typical team
            row["pos"][q] = {k: (((R[k] + shr * L[k] / L["min"]) / (R["min"] + shr)) / (L[k] / L["min"]) / (pace / lg_pace))
                             if L[k] else 1.0 for k in DEF_STATS}
        out[t] = row
    return {"teams": out, "pace": lg_pace, "positions": P}


def matchup_factors(dfn, team, opp, name=None):
    """{stat_key: multiplier} for a player on `team` facing `opp`: opponent defense (vs his position when known)
    x expected game pace."""
    T = dfn.get("teams", {})
    lg_pace = dfn.get("pace", 100.0)
    if opp not in T:
        return {k: 1.0 for k in DEF_STATS}
    tp = T.get(team, {}).get("pace", lg_pace)
    pace_f = _clamp((tp + T[opp]["pace"] - lg_pace) / tp, PACE_CAP)   # player's own rate already reflects his team's pace
    q = dfn.get("positions", {}).get(norm(name)) if name else None
    d = T[opp]["pos"].get(q) if q and T[opp].get("pos") else None
    return {k: _clamp(_clamp((d or T[opp])[k], DEF_CAP) * pace_f, MATCHUP_CAP) for k in DEF_STATS}


def injury_boosts(idx, injuries, pos=None):
    """Teammate injury boost. For every rotation player listed OUT, hand his minutes to healthy teammates
    (same position first), then hand USAGE_KEEP of whatever production those extra minutes do not replace
    (shots, rebounds, assists) to teammates in proportion to how much of that stat they already produce.
    Each boost is scaled by how often the injured player actually played in the teammate's recent games, so an
    absence that is already baked into the recent numbers is not counted twice. Uses saved box scores only."""
    P = {norm(k): v for k, v in (pos or {}).items()}
    out_by_team = {}
    for i in injuries:
        if i.get("status", "").lower() == "out":
            out_by_team.setdefault(i["team"], set()).add(norm(i["player"]))
    boosts = {}
    for team, outs in out_by_team.items():
        # team's recent games, newest first
        dates = sorted({r["date"] for rows in idx.values() for r in rows if r["t"] == team}, reverse=True)[:10]
        if len(dates) < 3:
            continue
        roster = {}
        for key, rows in idx.items():
            mine = [r for r in rows if r["t"] == team and r["date"] in dates and r["min"] >= 5]
            if mine:
                roster[key] = mine
        missing = []
        for o in outs:
            rows = roster.get(o)
            if not rows or len(rows) < 2:
                continue
            m = sum(r["min"] for r in rows) / len(rows)
            if m < 12:
                continue
            missing.append({"key": o, "name": rows[-1]["n"], "min": m, "dates": {r["date"] for r in rows},
                            "pg": {k: sum(r[k] for r in rows) / len(rows) for k in DEF_STATS}, "pos": P.get(o)})
        if not missing:
            continue
        active = {}
        for key, rows in roster.items():
            if key in outs or not any(r["date"] in dates[:5] for r in rows):
                continue
            m = sum(r["min"] for r in rows) / len(rows)
            if m < 10:
                continue
            tm = sum(r["min"] for r in rows)
            active[key] = {"min": m, "rate": {k: sum(r[k] for r in rows) / tm for k in DEF_STATS},
                           "pg": {k: sum(r[k] for r in rows) / len(rows) for k in DEF_STATS}, "pos": P.get(key),
                           "dates": [r["date"] for r in rows][-10:]}
        if not active:
            continue
        for O in missing:
            w = {b: a["min"] * (2.0 if O["pos"] and a["pos"] == O["pos"] else 1.0) for b, a in active.items()}
            W = sum(w.values())
            xmin = {b: O["min"] * w[b] / W for b in active}
            for b in active:                                       # nobody goes past 38 minutes or +8
                xmin[b] = max(0.0, min(xmin[b], 8.0, 38.0 - active[b]["min"]))
            for k in DEF_STATS:
                gap = max(0.0, O["pg"][k] - sum(xmin[b] * active[b]["rate"][k] for b in active)) * USAGE_KEEP
                tot = sum(a["pg"][k] for a in active.values())
                for b, a in active.items():
                    avail = sum(1 for d in a["dates"] if d in O["dates"]) / len(a["dates"])   # how often they shared the floor
                    if avail <= 0:
                        continue
                    B = boosts.setdefault(b, {"min": 0.0, "add": {x: 0.0 for x in DEF_STATS}, "out": []})
                    if k == DEF_STATS[0]:
                        B["min"] += xmin[b] * avail
                        B["out"].append(O["name"])
                    if tot:
                        B["add"][k] += gap * a["pg"][k] / tot * avail
    return boosts


SCHED_CAP = 0.05     # blowout risk + rest together can move a projection by at most +/-5%


def schedule_factor(mins, spread, rest, opp_rest):
    """Small, capped nudge for game script and fatigue. Returns (multiplier, [notes]).
    Blowout risk: past an 8-point spread, heavy-minute starters lose a little playing time (up to -4% at a
    16+ point spread) and deep bench players gain a little (up to +3%). Rest: own team on a back-to-back -3%,
    3+ days off +1%; opponent on a back-to-back +1.5% (tired defense)."""
    f, notes = 1.0, []
    if spread is not None:
        excess = max(0.0, abs(spread) - 8.0)
        if excess and mins >= 28:
            d = min(4.0, excess * 0.5)
            f *= 1 - d / 100
            notes.append(f"blowout risk ({abs(spread):g}-pt spread) -{d:g}%")
        elif excess and mins < 22:
            d = min(3.0, excess * 0.4)
            f *= 1 + d / 100
            notes.append(f"blowout minutes ({abs(spread):g}-pt spread) +{d:g}%")
    if rest == 0:
        f *= 0.97
        notes.append("back-to-back -3%")
    elif rest is not None and rest >= 3:
        f *= 1.01
        notes.append(f"{rest} days rest +1%")
    if opp_rest == 0:
        f *= 1.015
        notes.append("opponent on a back-to-back +1.5%")
    return _clamp(f, SCHED_CAP), notes


def project_player(games, dfn=None, opp=None, boost=None, sched=None):
    """games: chronological list of box-score rows. Returns {stat: projection} or None.

    Base = recent per-minute rate x projected minutes. With `dfn` (team_defense) and `opp`, the base rate is
    first neutralised for the defenses the player already faced, then scaled to tonight's opponent (by his
    position when known) and the expected pace of this game. `boost` (injury_boosts) adds minutes and usage
    when teammates are out. `sched` = {"spread", "rest", "opp_rest"} applies schedule_factor (blowout risk, rest).
    `_matchup` / `_injury` / `_sched` hold the % change from each."""
    recent = [g for g in games if g["min"] >= 5][-10:][::-1]   # newest first
    if len(recent) < 3:
        return None
    mins = wmean([g["min"] for g in recent[:8]], 0.85)
    if mins < 12:
        return None
    tot_min = sum(g["min"] for g in recent)
    rate = {k: sum(g[k] for g in recent) / tot_min for k in DEF_STATS}
    mx = {k: 1.0 for k in DEF_STATS}
    name = recent[0]["n"]
    if dfn and opp:
        team = recent[0]["t"]
        mx = matchup_factors(dfn, team, opp, name)
        faced = {k: sum(g["min"] * matchup_factors(dfn, g["t"], g.get("opp"), name)[k] for g in recent) / tot_min
                 for k in DEF_STATS}
        rate = {k: rate[k] / (faced[k] or 1) for k in DEF_STATS}
    xm = min((boost or {}).get("min", 0.0), max(0.0, 38.0 - mins))
    add = (boost or {}).get("add", {})
    base = {k: rate[k] * mins for k in DEF_STATS}
    inj = {k: min(xm * rate[k] + add.get(k, 0.0), base[k] * INJ_CAP) for k in DEF_STATS}
    raw = {s: sum(g[k] for g in recent) / tot_min * mins for s, k in STAT_KEY.items()}
    mid = {s: base[k] * mx[k] for s, k in STAT_KEY.items()}           # matchup only
    sf, snotes = schedule_factor(mins, sched.get("spread"), sched.get("rest"), sched.get("opp_rest")) if sched else (1.0, [])
    p = {s: (base[k] + inj[k]) * mx[k] * sf for s, k in STAT_KEY.items()}  # matchup + injuries + schedule
    for combo, parts in (("PRA", ("PTS", "REB", "AST")), ("PR", ("PTS", "REB")), ("PA", ("PTS", "AST")), ("RA", ("REB", "AST"))):
        for d in (p, raw, mid):
            d[combo] = sum(d[x] for x in parts)
    out = {k: rnd(v) for k, v in p.items()}
    out["_matchup"] = {k: rnd((mid[k] / raw[k] - 1) * 100, 1) if raw[k] else 0.0 for k in p}
    out["_injury"] = {k: rnd(((p[k] / sf) / mid[k] - 1) * 100, 1) if mid[k] else 0.0 for k in p}
    out["_sched"] = rnd((sf - 1) * 100, 1)
    out["_sched_notes"] = snotes
    return out


def odds_props(slate, plog_idx):
    """DraftKings player-prop lines via the-odds-api.com. Returns {event_id: [(player, stat, line)]}."""
    if not ODDS_KEY:
        return {}
    evs = CREDITS.spend(f"{ODDS}/events", 0, "event list")
    if not evs:
        return {}
    name_to_abbr = {}
    for g in slate:
        for ab, nm in g["names"].items():
            name_to_abbr[NAME_FIX.get(nm, nm)] = ab
    by_pair = {}
    for e in evs:
        h, a = name_to_abbr.get(NAME_FIX.get(e["home_team"], e["home_team"])), name_to_abbr.get(NAME_FIX.get(e["away_team"], e["away_team"]))
        if h and a:
            by_pair[(a, h)] = e["id"]
    out, used = {}, 0
    for g in slate:
        oid = by_pair.get((g["away"], g["home"]))
        if not oid or used >= MAX_PROP_EVENTS:
            continue
        used += 1
        j = CREDITS.spend(f"{ODDS}/events/{oid}/odds", len(PROP_MARKETS), f"props {g['away']}@{g['home']}",
                          regions="us", bookmakers="draftkings", markets=",".join(PROP_MARKETS), oddsFormat="american")
        found = {}
        for bk in (j or {}).get("bookmakers", []):
            for m in bk.get("markets", []):
                stat = PROP_MARKETS.get(m["key"])
                if not stat:
                    continue
                for o in m.get("outcomes", []):
                    if o.get("point") is None or o.get("name") not in ("Over", "Under"):
                        continue
                    key = (o.get("description", ""), stat)
                    row = found.setdefault(key, {"line": float(o["point"]), "over": None, "under": None})
                    if float(o["point"]) != row["line"]:
                        continue                      # keep the main (first) line only
                    row["over" if o["name"] == "Over" else "under"] = o.get("price")
        out[g["event_id"]] = [(pl, st, r["line"], r["over"], r["under"]) for (pl, st), r in found.items()]
    return out


def odds_game_lines():
    """DraftKings spread/total/moneyline for every NBA game in one call. {(away_name, home_name): lines}"""
    if not ODDS_KEY:
        return {}
    j = CREDITS.spend(f"{ODDS}/odds", 3, "game lines", regions="us", bookmakers="draftkings",
                      markets="h2h,spreads,totals", oddsFormat="american")
    out = {}
    for e in j or []:
        h, a = NAME_FIX.get(e["home_team"], e["home_team"]), NAME_FIX.get(e["away_team"], e["away_team"])
        row = {}
        for bk in e.get("bookmakers", []):
            for m in bk.get("markets", []):
                oc = {o.get("name"): o for o in m.get("outcomes", [])}
                if m["key"] == "spreads" and e["home_team"] in oc and oc[e["home_team"]].get("point") is not None:
                    row["spread"] = sgn(float(oc[e["home_team"]]["point"]))
                    row["spread_px_home"] = oc[e["home_team"]].get("price")
                    row["spread_px_away"] = (oc.get(e["away_team"]) or {}).get("price")
                elif m["key"] == "totals" and "Over" in oc and oc["Over"].get("point") is not None:
                    row["total"] = float(oc["Over"]["point"])
                    row["over_px"] = oc["Over"].get("price")
                    row["under_px"] = (oc.get("Under") or {}).get("price")
                elif m["key"] == "h2h":
                    fmt = lambda v: "N/A" if v is None else (f"+{v}" if v > 0 else str(v))
                    row["ml_home"] = fmt((oc.get(e["home_team"]) or {}).get("price"))
                    row["ml_away"] = fmt((oc.get(e["away_team"]) or {}).get("price"))
        if "spread" in row and "total" in row:
            row.setdefault("ml_home", "N/A"); row.setdefault("ml_away", "N/A")
            out[(a, h)] = row
    return out


def fetch_standings():
    j = get("https://site.api.espn.com/apis/v2/sports/basketball/nba/standings")
    if not j:
        return None
    confs = []
    for c in j.get("children", []):
        rows = []
        for e in (c.get("standings") or {}).get("entries", []):
            st = {s.get("name"): s for s in e.get("stats", [])}
            dv = lambda k: (st.get(k) or {}).get("displayValue", "")
            rows.append({"team": abbr(e["team"].get("abbreviation")), "name": e["team"].get("displayName", ""),
                         "w": int(num((st.get("wins") or {}).get("value"))), "l": int(num((st.get("losses") or {}).get("value"))),
                         "pct": dv("winPercent"), "gb": dv("gamesBehind"), "seed": int(num((st.get("playoffSeed") or {}).get("value"))),
                         "streak": dv("streak"), "l10": dv("Last Ten Games"), "home": dv("Home"), "road": dv("Road"),
                         "ppg": dv("avgPointsFor"), "opp": dv("avgPointsAgainst"), "diff": dv("differential")})
        rows.sort(key=lambda r: (r["seed"] or 99, -r["w"]))
        confs.append({"name": c.get("name", ""), "teams": rows})
    return confs or None


def fetch_futures():
    """NBA championship odds. DraftKings when it is listed, otherwise the first US book."""
    if not ODDS_KEY:
        return None
    j = CREDITS.spend("https://api.the-odds-api.com/v4/sports/basketball_nba_championship_winner/odds", 1,
                      "championship odds", regions="us", markets="outrights", oddsFormat="american")
    books = [bk for e in (j or []) for bk in e.get("bookmakers", [])]
    log(f"futures: {len(books)} books ({', '.join(b.get('key', '') for b in books[:6])})")
    if not books:
        return None
    bk = next((b for b in books if b.get("key") == "draftkings"), books[0])
    rows = []
    for m in bk.get("markets", []):
        for o in m.get("outcomes", []):
            nm = NAME_FIX.get(o.get("name"), o.get("name"))
            ab = next((k for k, v in TEAM_NAMES.items() if v == nm), "")
            price = o.get("price")
            if price is None:
                continue
            implied = 100 / (price + 100) if price > 0 else -price / (-price + 100)
            rows.append({"team": ab, "name": nm, "odds": f"+{price}" if price > 0 else str(price), "implied": rnd(implied * 100, 1)})
    rows.sort(key=lambda r: -r["implied"])
    if rows:
        rows[0]["book"] = bk.get("title", bk.get("key", ""))
    return rows or None


def team_rest(plog, slate):
    """Days of rest before each slate game, from the finished-game log."""
    played = {}
    for g in plog["games"].values():
        for t in (g["home"], g["away"]):
            played.setdefault(t, set()).add(g["date"])
    out = {}
    for g in slate:
        d = date.fromisoformat(g["date"])
        rest = {}
        for side in ("away", "home"):
            prior = [x for x in played.get(g[side], ()) if x < g["date"]]
            rest[side] = (d - date.fromisoformat(max(prior))).days - 1 if prior else None
        out[g["event_id"]] = rest
    return out


GL_COLS = ["date", "opp", "home", "min", "pts", "reb", "ast", "tpm", "stl", "blk", "fgm", "fga",
           "tpa", "ftm", "fta", "to", "oreb", "dreb", "pf", "pm"]


def build_game_logs(idx, slate, stamp):
    """Last 15 games for every rotation player on a team in the active slate."""
    teams = {g["away"] for g in slate if g["status"] != "Final"} | {g["home"] for g in slate if g["status"] != "Final"}
    players = {}
    for games in idx.values():
        if games[-1]["t"] not in teams:
            continue
        played = [g for g in games if g["min"] >= 1]
        last5 = played[-5:]
        if len(played) < 3 or sum(g["min"] for g in last5) / len(last5) < 12:
            continue
        rows = [[(1 if g[c] is True else 0 if g[c] is False else g[c]) for c in GL_COLS] for g in played[-15:][::-1]]
        players[games[-1]["n"]] = {"team": games[-1]["t"], "rows": rows}
    return {"updated": stamp, "cols": GL_COLS, "players": players}


def fetch_injuries(slate):
    j = get(f"{ESPN}/injuries")
    if not j:
        return None
    team_game = {}
    for g in slate:
        if g["status"] != "Final":
            for ab, nm in g["names"].items():
                team_game[nm] = (ab, g["id"])
    rows = []
    for t in j.get("injuries", []):
        hit = team_game.get(t.get("displayName"))
        if not hit:
            continue
        for i in t.get("injuries", []):
            det = i.get("details") or {}
            reason = " ".join(x for x in (det.get("side"), det.get("type") or det.get("location")) if x)
            if not reason:
                reason = (i.get("shortComment") or "").strip()[:80]
            rows.append({"player": (i.get("athlete") or {}).get("displayName", ""), "team": hit[0],
                         "status": i.get("status", ""), "reason": reason or "Undisclosed", "game": hit[1]})
    return rows


# -------------------------------------------------------------- settlement
STAT_VAL = {"PTS": lambda r: r["pts"], "REB": lambda r: r["reb"], "AST": lambda r: r["ast"], "3PM": lambda r: r["tpm"],
            "BLK": lambda r: r["blk"], "STL": lambda r: r["stl"], "PRA": lambda r: r["pts"] + r["reb"] + r["ast"],
            "PR": lambda r: r["pts"] + r["reb"], "PA": lambda r: r["pts"] + r["ast"], "RA": lambda r: r["reb"] + r["ast"]}


def units_for(res, price):
    if res == "W":
        return rnd(price / 100 if price and price > 0 else 100 / -price if price else WIN_UNITS, 2)
    return -1.0 if res == "L" else 0.0


def implied(price):
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def ml_num(v):
    try:
        return int(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None


def grade_potd(history, plog):
    """Grade any Pick of the Day whose game has finished (props, spreads, totals and moneylines)."""
    for h in history:
        if h.get("result") or h["eid"] not in plog["games"]:
            continue
        g = plog["games"][h["eid"]]
        kind = h.get("type", "Prop")
        if kind == "Prop":
            row = next((r for r in g["players"] if norm(r["n"]) == norm(h["player"])), None)
            if row is None:
                h["result"], h["actual"], h["units"] = "P", "Did not play (void)", 0.0
                continue
            act = STAT_VAL[h["stat"]](row)
            over = h["side"] == "Over"
            h["result"] = "P" if act == h["line"] else ("W" if (act > h["line"]) == over else "L")
            h["actual"] = f"{act:g} {h['stat']}"
        else:
            pts = {g["home"]: g["home_pts"], g["away"]: g["away_pts"]}
            score = f"{g['away']} {g['away_pts']} - {g['home_pts']} {g['home']}"
            if kind == "Total":
                tot = g["home_pts"] + g["away_pts"]
                h["result"] = "P" if tot == h["line"] else ("W" if (tot > h["line"]) == (h["side"] == "Over") else "L")
                h["actual"] = f"Total {tot} ({score})"
            else:
                t = h["team"]
                opp = g["home"] if t == g["away"] else g["away"]
                margin = pts[t] - pts[opp]
                if kind == "Spread":
                    cover = margin + h["line"]
                    h["result"] = "P" if cover == 0 else ("W" if cover > 0 else "L")
                else:
                    h["result"] = "W" if margin > 0 else "L"
                h["actual"] = score
        h["units"] = units_for(h["result"], h.get("price"))
        h["graded_at"] = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p ET")


def team_log(plog):
    """team -> list of (date, points for, points against), oldest first."""
    out = {}
    for g in sorted(plog["games"].values(), key=lambda x: x["date"]):
        out.setdefault(g["home"], []).append((g["date"], g["home_pts"], g["away_pts"]))
        out.setdefault(g["away"], []).append((g["date"], g["away_pts"], g["home_pts"]))
    return out


def _rate(vals, ok, n):
    sl = vals[:n]
    return (sum(1 for v in sl if ok(v)), len(sl)) if sl else (0, 0)


COMBO_KEYS = {"PTS": ("pts",), "REB": ("reb",), "AST": ("ast",), "3PM": ("tpm",), "BLK": ("blk",), "STL": ("stl",),
              "PRA": ("pts", "reb", "ast"), "PR": ("pts", "reb"), "PA": ("pts", "ast"), "RA": ("reb", "ast")}
TIER_NAME = {"tough": "top-10 defenses", "mid": "middle-10 defenses", "soft": "bottom-10 defenses"}


def defense_tiers(dfn, stat, q):
    """{team: 'tough'|'mid'|'soft'} for one prop stat against position q (team-level when q is unknown).
    Ranked by what each team allows; top 10 stingiest are 'tough', bottom 10 'soft'."""
    T = dfn.get("teams", {})
    keys = COMBO_KEYS.get(stat, ())
    if not T or not keys:
        return {}
    val = {}
    for t, r in T.items():
        src = (r.get("pos") or {}).get(q) if q else None
        src = src or r
        val[t] = sum(src[k] for k in keys) / len(keys)
    order = sorted(val, key=val.get)
    n = len(order)
    return {t: ("tough" if i < n / 3 else "soft" if i >= n - n / 3 else "mid") for i, t in enumerate(order)}


def add_tier_hits(props, idx, dfn):
    """#5 Matchup-aware hit rates: how the player did against defenses like tonight's (same third of the
    league in that stat vs his position), plus vs the toughest and softest thirds. Display + reasons only."""
    cache = {}
    for p in props:
        if p["stat"] not in STAT_VAL or not p.get("opp"):
            continue
        q = dfn.get("positions", {}).get(norm(p["player"]))
        ck = (p["stat"], q)
        if ck not in cache:
            cache[ck] = defense_tiers(dfn, p["stat"], q)
        tiers = cache[ck]
        if p["opp"] not in tiers:
            continue
        rows = [r for r in idx.get(norm(p["player"]), []) if r["min"] >= 5 and r.get("opp") in tiers]
        over = p.get("side", "Over") == "Over"
        res = {"tonight": tiers[p["opp"]]}
        for tier in ("tough", "mid", "soft"):
            vals = [STAT_VAL[p["stat"]](r) for r in rows if tiers[r["opp"]] == tier]
            hit = sum(1 for v in vals if (v > p["dk"] if over else v < p["dk"]))
            res[tier] = [hit, len(vals)]
        p["tiers"] = res


def tier_note(p):
    t = p.get("tiers")
    if not t:
        return None
    a, n = t[t["tonight"]]
    side = "over" if p["edge"] > 0 else "under"
    if not n:
        return f"{p['opp']} ranks among the {TIER_NAME[t['tonight']]} for {p['stat']}; no games against that tier yet"
    return f"{p['opp']} ranks among the {TIER_NAME[t['tonight']]} for {p['stat']}; he went {side} {p['dk']:g} in {a} of {n} games against that tier"


def market_now(bet, games, props):
    """The current DraftKings line and price for a saved bet (from this run's data, no extra API calls)."""
    if bet.get("type") == "Prop":
        p = next((x for x in props if x["game"] == bet["game"] and x["stat"] == bet.get("stat")
                  and norm(x["player"]) == norm(bet.get("player", ""))), None)
        if not p:
            return None
        px = p.get("over") if bet["side"] == "Over" else p.get("under")
        return (p["dk"], px) if px is not None else None
    row = next((r for r in games if r["id"] == bet["game"] and r["date"] == bet["date"]), None)
    dk = (row or {}).get("dk") or {}
    try:
        if bet["type"] == "Spread":
            hs = float(dk["spread"])
            home = bet["team"] == row["home"]
            return (hs if home else -hs, dk.get("spread_px_home" if home else "spread_px_away") or -110)
        if bet["type"] == "Total":
            return (float(dk["total"]), dk.get("over_px" if bet["side"] == "Over" else "under_px") or -110)
        if bet["type"] == "Moneyline":
            px = ml_num(dk.get("ml_home" if bet["team"] == row["home"] else "ml_away"))
            return (0, px) if px is not None else None
    except (KeyError, TypeError, ValueError):
        return None
    return None


def track_clv(entries, games, props, slate, now):
    """#6 Closing line value. While a bet's game has not tipped, keep copying the latest DraftKings line into
    close_line / close_price; once it tips that is the closing number. clv.line > 0 means the line moved our
    way (e.g. our Over 22.5 closed at 23.5); clv.price > 0 means the price got more expensive after we
    took it, in points of implied probability. Beating the close over time is the best sign of real edges."""
    status = {g["event_id"]: g["status"] for g in slate}
    for e in entries:
        if e.get("result") or not e.get("price"):
            continue
        e.setdefault("open_line", e.get("line"))
        e.setdefault("open_price", e.get("price"))
        if status.get(e.get("eid")) == "Upcoming":
            cur = market_now(e, games, props)
            if cur:
                e["close_line"], e["close_price"] = cur
                e["close_at"] = now.strftime("%Y-%m-%d %I:%M %p ET")
        if e.get("close_price") is None:
            continue
        ol, cl = e.get("open_line") or 0, e.get("close_line") or 0
        if e["type"] == "Moneyline":
            lm = 0.0
        elif e["type"] == "Spread":
            lm = ol - cl                                    # we hold +7.5 and it closes +6.5: beat it by 1
        else:
            lm = (cl - ol) if e["side"] == "Over" else (ol - cl)
        pm = (implied(e["close_price"]) - implied(e["open_price"])) * 100
        e["clv"] = {"line": rnd(lm, 1), "price": rnd(pm, 1), "beat": bool(lm > 0 or (lm == 0 and pm > 0.5)),
                    "final": status.get(e.get("eid")) != "Upcoming"}


def clv_summary(entries):
    done = [e for e in entries if e.get("clv") and e["clv"].get("final")]
    if not done:
        return {"bets": 0}
    return {"bets": len(done), "beat": sum(1 for e in done if e["clv"]["beat"]),
            "avg_line": rnd(sum(e["clv"]["line"] for e in done) / len(done), 2),
            "avg_price": rnd(sum(e["clv"]["price"] for e in done) / len(done), 2)}


def carry_open(new, old):
    """When a re-pick lands on the same bet, keep the first price we posted so CLV is measured from it."""
    if new and old and new.get("label") == old.get("label"):
        for k in ("open_line", "open_price", "first_picked_at"):
            if old.get(k) is not None:
                new[k] = old[k]
        new.setdefault("first_picked_at", old.get("first_picked_at") or old.get("picked_at"))
    return new


def sched_ctx(row, team):
    """Spread (DraftKings when posted, else the model) and rest days for `team` in this game row."""
    dk = (row or {}).get("dk") or {}
    try:
        spread = float(dk.get("spread"))
    except (TypeError, ValueError):
        spread = (row or {}).get("model", {}).get("spread")
    rest = (row or {}).get("rest") or {}
    side = "home" if team == (row or {}).get("home") else "away"
    return {"spread": spread, "rest": rest.get(side), "opp_rest": rest.get("away" if side == "home" else "home")}


def sched_note(p):
    return "Game script / rest: " + ", ".join(p.get("sched_notes") or []) + f" (net {'+' if p['sched'] > 0 else ''}{p['sched']:g}%)"


def matchup_note(p):
    m, opp = p.get("matchup"), p.get("opp")
    if m is None or not opp:
        return "Matchup: not adjusted"
    if abs(m) < 1:
        return f"Matchup vs {opp}: neutral for {p['stat']} (defense and pace about league average)"
    good = (m > 0) == (p["edge"] > 0)
    return (f"Matchup vs {opp}: {'+' if m > 0 else ''}{m:g}% to his {p['stat']} projection from their defense and pace"
            f" ({'helps' if good else 'works against'} the {'over' if p['edge'] > 0 else 'under'})")


def injury_note(p):
    who = ", ".join(p.get("inj_out") or []) or "teammates"
    m = p["injury"]
    return f"Teammate injury boost: {who} out, {'+' if m > 0 else ''}{m:g}% to his {p['stat']} projection (extra minutes and usage)"


def pick_of_the_day(props, games, slate, idx, plog, injuries, today_iso):
    """Rank every bet on today's board (player props, spreads, totals, moneylines). Returns (pick, runners_up).

    Every candidate needs real support from recent results, a sane edge (very large edges usually mean the
    model is missing news) and a playable price. Score = 55% recent hit rate + 45% edge size, on a 0-100 scale.
    """
    ev = {g["id"]: g for g in slate}
    hurt = {norm(i["player"]): i["status"] for i in injuries}
    tl = team_log(plog)
    cands = []

    def base(g, row):
        return {"date": today_iso, "eid": g["event_id"], "game": row["id"], "time": row["time"],
                "home": row["home"], "away": row["away"]}

    def score_of(h5, h10, h15, edge_part):
        return rnd((0.55 * (0.5 * h10 + 0.25 * h5 + 0.25 * h15) + 0.45 * min(edge_part, 1)) * 100, 1)

    # ---- player props
    for p in props:
        g = ev.get(p["game"])
        if not g or g["status"] != "Upcoming" or g["date"] != today_iso or not p["dk"]:
            continue
        if norm(p["player"]) in hurt or p["stat"] not in STAT_VAL:
            continue
        over = p["edge"] > 0
        price = p.get("over") if over else p.get("under")
        if price is None or price < -160:
            continue
        hist = [r for r in idx.get(norm(p["player"]), []) if r["min"] >= 5]
        if len(hist) < 8:
            continue
        rs = hist[::-1]
        vals = [STAT_VAL[p["stat"]](r) for r in rs]
        ok = (lambda v: v > p["dk"]) if over else (lambda v: v < p["dk"])
        (a5, n5), (a10, n10), (a15, n15) = _rate(vals, ok, 5), _rate(vals, ok, 10), _rate(vals, ok, 15)
        epct = abs(p["edge"]) / p["dk"]
        m5 = sum(r["min"] for r in rs[:5]) / n5
        m15 = sum(r["min"] for r in rs[:15]) / n15
        # Very large edges usually mean the model is missing news. A teammate injury boost IS that news, so the
        # part of the edge it explains does not count against the 40% ceiling.
        inj = (p.get("injury") or 0) / 100 if over else 0.0
        if a10 / n10 < 0.6 or not (0.06 <= epct <= 0.40 + inj) or m5 < 20 or m5 < m15 - 6:
            continue
        side = "Over" if over else "Under"
        row = next(r for r in games if r["id"] == p["game"])
        cands.append({**base(g, row), "type": "Prop", "player": p["player"], "team": p["team"], "stat": p["stat"],
                      "side": side, "line": p["dk"], "price": price, "model": p["model"], "edge": p["edge"],
                      "edge_pct": rnd(epct * 100, 1), "label": f"{p['player']} {side} {p['dk']:g} {p['stat']}",
                      "hits": {"l5": [a5, n5], "l10": [a10, n10], "l15": [a15, n15]},
                      "chart": {"vals": vals[:10][::-1], "line": p["dk"], "side": side, "label": p["stat"]},
                      "score": score_of(a5 / n5, a10 / n10, a15 / n15, epct / 0.25),
                      "reasons": [f"Model projects {p['model']:g} {p['stat']} vs a line of {p['dk']:g} ({'+' if p['edge'] > 0 else ''}{p['edge']:g}, {round(epct * 100)}% edge)",
                                  f"Went {side.lower()} the line in {a10} of his last {n10} games ({a5}/{n5} in the last 5)",
                                  f"Averaging {sum(vals[:n5]) / n5:.1f} over the last 5 and {sum(vals[:n15]) / n15:.1f} over the last {n15}",
                                  f"Minutes steady: {m5:.0f} per game over the last 5 ({m15:.0f} over the last {n15})",
                                  matchup_note(p),
                                  *([tier_note(p)] if p.get("tiers") else []),
                                  *([injury_note(p)] if p.get("injury") else []),
                                  *([sched_note(p)] if abs(p.get("sched") or 0) >= 1 else []),
                                  "Not on the injury report at pick time"]})

    # ---- game bets
    for row in games:
        g = ev.get(row["id"])
        dk = row.get("dk") or {}
        if not g or g["status"] != "Upcoming" or g["date"] != today_iso or dk.get("spread") in (None, "N/A") or not dk.get("total"):
            continue
        H, A = row["home"], row["away"]
        th, ta = tl.get(H, [])[::-1], tl.get(A, [])[::-1]            # newest first
        if len(th) < 6 or len(ta) < 6:
            continue
        rest = row.get("rest") or {}
        rest_txt = []
        for t, side_key in ((A, "away"), (H, "home")):
            d = rest.get(side_key)
            if d == 0:
                rest_txt.append(f"{t} on a back-to-back")
            elif d is not None and d >= 2:
                rest_txt.append(f"{t} on {d} days rest")
        model_margin = -row["model"]["spread"]                       # home perspective
        hs = float(dk["spread"])

        # spread
        se = row.get("spread_edge", 0)
        if 1.5 <= abs(se) <= 6:
            home_pick = se > 0
            team, opp = (H, A) if home_pick else (A, H)
            line = hs if home_pick else -hs
            price = dk.get("spread_px_home" if home_pick else "spread_px_away") or -110
            margins = [f - a for _, f, a in (th if home_pick else ta)]
            ok = lambda m: m + line > 0
            (a5, n5), (a10, n10), (a15, n15) = _rate(margins, ok, 5), _rate(margins, ok, 10), _rate(margins, ok, 15)
            if a10 / n10 >= 0.6 and price >= -160:
                mm = model_margin if home_pick else -model_margin
                cands.append({**base(g, row), "type": "Spread", "team": team, "side": "Cover", "line": line, "price": price,
                              "model": rnd(mm), "edge": rnd(abs(se)), "edge_pct": None, "label": f"{team} {sgn(line)}",
                              "hits": {"l5": [a5, n5], "l10": [a10, n10], "l15": [a15, n15]},
                              "chart": {"vals": margins[:10][::-1], "line": -line, "side": "Over", "label": f"{team} margin"},
                              "score": score_of(a5 / n5, a10 / n10, a15 / n15, abs(se) / 4),
                              "reasons": [f"Model has {team} {'winning' if mm > 0 else 'losing'} by {abs(mm):.1f}; the line is {team} {sgn(line)} ({abs(se):g}-point edge)",
                                          f"{team} would have covered {sgn(line)} in {a10} of their last {n10} games ({a5}/{n5} in the last 5)",
                                          f"{team} average margin: {sum(margins[:n5]) / n5:+.1f} over the last 5, {sum(margins[:n15]) / n15:+.1f} over the last {n15}"] + rest_txt})

        # total
        te = row.get("total_edge", 0)
        if 3 <= abs(te) <= 12:
            over = te > 0
            line = float(dk["total"])
            price = dk.get("over_px" if over else "under_px") or -110
            tot = [f + a for _, f, a in th]
            tot2 = [f + a for _, f, a in ta]
            ok = (lambda v: v > line) if over else (lambda v: v < line)
            def both(n):
                a1, n1 = _rate(tot, ok, n); a2, n2 = _rate(tot2, ok, n)
                return a1 + a2, n1 + n2
            (a5, n5), (a10, n10), (a15, n15) = both(5), both(10), both(15)
            if a10 / n10 >= 0.6 and price >= -160:
                side = "Over" if over else "Under"
                mixed = [round((x + y) / 2, 1) for x, y in zip(tot[:10], tot2[:10])][::-1]
                cands.append({**base(g, row), "type": "Total", "team": H, "side": side, "line": line, "price": price,
                              "model": row["model"]["total"], "edge": rnd(abs(te)), "edge_pct": None, "label": f"{A} @ {H} {side} {line:g}",
                              "hits": {"l5": [a5, n5], "l10": [a10, n10], "l15": [a15, n15]},
                              "chart": {"vals": mixed, "line": line, "side": side, "label": "avg game total, both teams"},
                              "score": score_of(a5 / n5, a10 / n10, a15 / n15, abs(te) / 8),
                              "reasons": [f"Model total {row['model']['total']:g} vs a line of {line:g} ({abs(te):g}-point edge)",
                                          f"{a10} of the two teams' last {n10} games went {side.lower()} {line:g}",
                                          f"{H} games average {sum(tot[:5]) / len(tot[:5]):.1f} total points over the last 5; {A} games {sum(tot2[:5]) / len(tot2[:5]):.1f}"] + rest_txt})

        # moneyline
        for team, opp, p_win, px in ((H, A, row["model"]["ml_home"], ml_num(dk.get("ml_home"))),
                                     (A, H, row["model"]["ml_away"], ml_num(dk.get("ml_away")))):
            if px is None or not (-250 <= px <= 250):
                continue
            imp = implied(px)
            edge = p_win - imp
            if not (0.04 <= edge <= 0.20):
                continue
            res = th if team == H else ta
            wins = [f - a for _, f, a in res]
            ok = lambda m: m > 0
            (a5, n5), (a10, n10), (a15, n15) = _rate(wins, ok, 5), _rate(wins, ok, 10), _rate(wins, ok, 15)
            if a10 / n10 < 0.5:
                continue
            evu = p_win * (px / 100 if px > 0 else 100 / -px) - (1 - p_win)
            cands.append({**base(g, row), "type": "Moneyline", "team": team, "side": "Win", "line": 0, "price": px,
                          "model": rnd(p_win * 100, 1), "edge": rnd(edge * 100, 1), "edge_pct": None, "label": f"{team} ML",
                          "hits": {"l5": [a5, n5], "l10": [a10, n10], "l15": [a15, n15]},
                          "chart": {"vals": wins[:10][::-1], "line": 0, "side": "Over", "label": f"{team} margin"},
                          "score": score_of(a5 / n5, a10 / n10, a15 / n15, edge / 0.10),
                          "reasons": [f"Model gives {team} a {p_win * 100:.0f}% chance to win; {sgn(px)} implies {imp * 100:.0f}% ({edge * 100:.1f}-point edge)",
                                      f"Expected value: {evu * 100:+.1f}% per unit at {sgn(px)}",
                                      f"{team} won {a10} of their last {n10} ({a5}/{n5} in the last 5)"] + rest_txt})

    cands.sort(key=lambda c: -c["score"])
    for c in cands:
        c["candidates_considered"] = len(cands)
    pick = cands[0] if cands else None
    key = lambda c: c.get("player") or c["game"] + c["type"]
    seen, alts = {key(pick) if pick else None}, []
    for c in cands[1:]:
        if key(c) in seen:
            continue
        seen.add(key(c)); alts.append(c)
        if len(alts) == 6:
            break
    return pick, alts, cands


# ------------------------------------------------------------------ win probability
def poisson_over(lam, line):
    """P(X > line) for Poisson(lam); line is a half number like 2.5."""
    k = int(math.floor(line))
    if lam <= 0:
        return 0.0
    term, cdf = math.exp(-lam), math.exp(-lam)
    for i in range(1, k + 1):
        term *= lam / i
        cdf += term
    return max(0.0, min(1.0, 1 - cdf))


COUNT_STATS = {"REB", "AST", "3PM", "BLK", "STL"}


def prob_over(stat, mean, line, vals=None):
    if stat in COUNT_STATS:
        return poisson_over(mean, line)
    sd = 1.0
    if vals and len(vals) >= 3:
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    sd = max(sd, 0.25 * mean, 2.0)            # single-game scoring is noisy; never trust a tiny sample sd
    return 1 - phi((line - mean) / sd)


def add_win_probs(cands):
    """Attach p_win: 60% model distribution + 40% recent hit rate (shrinks small samples toward the model)."""
    for c in cands:
        h = c["hits"]["l15"] if c["type"] == "Prop" else c["hits"]["l10"]
        hr = h[0] / h[1] if h[1] else 0.5
        if c["type"] == "Prop":
            po = prob_over(c["stat"], c["model"], c["line"], c["chart"]["vals"])
            pm = po if c["side"] == "Over" else 1 - po
        elif c["type"] == "Spread":
            pm = phi((c["model"] + c["line"]) / GAME_SD)
        elif c["type"] == "Total":
            po = 1 - phi((c["line"] - c["model"]) / 17.0)
            pm = po if c["side"] == "Over" else 1 - po
        else:
            pm = c["model"] / 100
        c["p_model"] = rnd(pm * 100, 1)
        c["p_win"] = rnd((0.6 * pm + 0.4 * hr) * 100, 1)
        dec = 1 + (c["price"] / 100 if c["price"] > 0 else 100 / -c["price"])
        c["ev"] = rnd((c["p_win"] / 100 * dec - 1) * 100, 1)


def decimal(price):
    return 1 + (price / 100 if price > 0 else 100 / -price)


# ------------------------------------------------------------------ ladder challenge
def ladder_pick(cands):
    """Safest bet with a real payout: price -200 to +120, positive expected value. Highest win chance wins."""
    pool = [c for c in cands if -200 <= c["price"] <= 120 and c.get("ev", -1) > 0]
    pool.sort(key=lambda c: (-c["p_win"], -c["ev"]))
    return pool[0] if pool else None


def run_ladder(lad, plog, cands, slate, t_iso, now):
    lad.setdefault("start", LADDER_START); lad.setdefault("goal", LADDER_GOAL)
    lad.setdefault("attempt", 1); lad.setdefault("bankroll", lad["start"]); lad.setdefault("steps", [])
    lad.setdefault("attempts", []); lad.setdefault("attempt_started", t_iso)
    stamp = now.strftime("%Y-%m-%d %I:%M %p ET")
    # 1. grade finished steps, oldest first, and move the bankroll
    grade_potd(lad["steps"], plog)
    stale = (now.date() - timedelta(days=3)).isoformat()
    for st in lad["steps"]:
        if not st.get("result") and st["date"] < stale:           # postponed / never graded: void it
            st.update(result="P", actual="No result (void)", units=0.0)
    for st in sorted(lad["steps"], key=lambda x: x["date"]):
        if not st.get("result") or st.get("applied"):
            continue
        st["applied"] = True
        if st["result"] == "W":
            lad["bankroll"] = rnd(st["stake"] * decimal(st["price"]), 2)
            st["bankroll_after"] = lad["bankroll"]
            if lad["bankroll"] >= lad["goal"]:
                lad["attempts"].insert(0, {"attempt": st["attempt"], "start": lad["attempt_started"], "end": st["date"],
                                           "days": st["day"], "peak": lad["bankroll"], "result": "Completed"})
                lad["attempt"] += 1; lad["bankroll"] = lad["start"]; lad["attempt_started"] = None
        elif st["result"] == "L":
            st["bankroll_after"] = 0
            peak = max([x.get("bankroll_after", 0) for x in lad["steps"] if x["attempt"] == st["attempt"]] + [lad["start"]])
            lad["attempts"].insert(0, {"attempt": st["attempt"], "start": lad["attempt_started"], "end": st["date"],
                                       "days": st["day"], "peak": peak, "result": "Busted"})
            lad["attempt"] += 1; lad["bankroll"] = lad["start"]; lad["attempt_started"] = None
        else:
            lad["bankroll"] = st["stake"]; st["bankroll_after"] = st["stake"]
    # 2. today's step: re-picked until its game tips off, then locked
    today = next((x for x in lad["steps"] if x["date"] == t_iso), None)
    pending_old = any(not x.get("result") for x in lad["steps"] if x["date"] < t_iso)
    if pending_old:
        log("ladder: waiting on an earlier step to be graded")
        return lad
    locked = bool(today) and bool(today.get("result") or next((g["status"] != "Upcoming" for g in slate if g["event_id"] == today["eid"]), True))
    if today and locked:
        today.setdefault("locked_at", stamp)
        return lad
    lad["steps"] = [x for x in lad["steps"] if x["date"] != t_iso]
    pick = ladder_pick(cands)
    prev_step = today
    if pick:
        if not lad.get("attempt_started"):
            lad["attempt_started"] = t_iso
        day = 1 + sum(1 for x in lad["steps"] if x["attempt"] == lad["attempt"])
        stake = lad["bankroll"]
        step = {k: pick[k] for k in ("type", "label", "game", "eid", "time", "home", "away", "side", "line", "price",
                                     "model", "edge", "hits", "reasons", "p_win", "p_model", "ev", "score")
                if k in pick}
        for k in ("player", "team", "stat"):
            if k in pick:
                step[k] = pick[k]
        step.update({"date": t_iso, "attempt": lad["attempt"], "day": day, "stake": stake,
                     "to_win": rnd(stake * (decimal(pick["price"]) - 1), 2), "picked_at": stamp,
                     "why_ladder": f"Highest win chance on the board at a ladder-friendly price: {pick['p_win']:g}% to win, {pick['ev']:+g}% expected value"})
        lad["steps"].insert(0, carry_open(step, prev_step))
        log(f"ladder: attempt {lad['attempt']} day {day}, ${stake:g} on {pick['label']} ({fmt_px(pick['price'])})")
    else:
        log("ladder: no bet cleared the ladder filters today (rest day)")
    return lad


def fmt_px(p):
    return f"+{p}" if p > 0 else str(p)


# ------------------------------------------------------------------ alt-line ladders
def odds_alt_lines(games_for_alt, slate):
    """Alternate player lines (PTS/REB/AST/3PM) for a few games. {event_id: {(player, stat): [(line, over_price)]}}"""
    if not ODDS_KEY or not games_for_alt:
        return {}
    evs = CREDITS.spend(f"{ODDS}/events", 0, "event list")
    if not evs:
        return {}
    name_to_abbr = {NAME_FIX.get(nm, nm): ab for g in slate for ab, nm in g["names"].items()}
    by_pair = {}
    for e in evs:
        h = name_to_abbr.get(NAME_FIX.get(e["home_team"], e["home_team"]))
        a = name_to_abbr.get(NAME_FIX.get(e["away_team"], e["away_team"]))
        if h and a:
            by_pair[(a, h)] = e["id"]
    out = {}
    for g in games_for_alt:
        oid = by_pair.get((g["away"], g["home"]))
        if not oid:
            continue
        j = CREDITS.spend(f"{ODDS}/events/{oid}/odds", len(ALT_MARKETS), f"alt lines {g['away']}@{g['home']}",
                          regions="us", bookmakers="draftkings", markets=",".join(ALT_MARKETS), oddsFormat="american")
        rows = {}
        for bk in (j or {}).get("bookmakers", []):
            for m in bk.get("markets", []):
                stat = ALT_MARKETS.get(m["key"])
                for o in m.get("outcomes", []):
                    if stat and o.get("name") == "Over" and o.get("point") is not None and o.get("price") is not None:
                        rows.setdefault((o.get("description", ""), stat), {})[float(o["point"])] = o["price"]
        out[g["event_id"]] = {k: sorted(v.items()) for k, v in rows.items()}
    return out


def build_alt_ladders(alt, slate, idx, props, injuries):
    gm = {g["event_id"]: g for g in slate}
    hurt = {norm(i["player"]) for i in injuries if i["status"].lower() == "out"}
    proj = {(p["player"], p["stat"]): p for p in props}
    out = []
    for eid, rows in alt.items():
        g = gm.get(eid)
        for (player, stat), rungs in rows.items():
            hist = [r for r in idx.get(norm(player), []) if r["min"] >= 5]
            if norm(player) in hurt or len(hist) < 5 or len(rungs) < 3:
                continue
            vals = [STAT_VAL[stat](r) for r in hist[::-1]][:15]
            pr = next((v for (pl, st), v in proj.items() if norm(pl) == norm(player) and st == stat), None)
            mean = pr["model"] if pr else sum(vals[:10]) / len(vals[:10])
            ladder = []
            for line, price in rungs:
                if price < -1000 or price > 3000:
                    continue
                pm = prob_over(stat, mean, line, vals)
                hit = sum(1 for v in vals if v > line)
                p = 0.6 * pm + 0.4 * hit / len(vals)
                imp = implied(price)
                ladder.append({"line": line, "price": price, "implied": rnd(imp * 100, 1), "p": rnd(p * 100, 1),
                               "hit": [hit, len(vals)], "ev": rnd((p * decimal(price) - 1) * 100, 1)})
            ladder = [r for r in ladder if 1 <= r["implied"] <= 95]
            if len(ladder) < 3:
                continue
            # suggested 3-rung ladder: safest +EV-ish rung, a middle rung, and a long shot with value
            value = [r for r in ladder if r["ev"] > 0 and -250 <= r["price"] <= 1000]
            pick = []
            if value:
                safe = max(value, key=lambda r: r["p"])
                longs = [r for r in value if r["price"] >= 200 and r["line"] > safe["line"]]
                mids = [r for r in value if safe["price"] < r["price"] < 200 and r["line"] > safe["line"]]
                pick = [safe] + ([max(mids, key=lambda r: r["ev"])] if mids else []) + ([max(longs, key=lambda r: r["ev"])] if longs else [])
            best = max(ladder, key=lambda r: r["ev"])
            out.append({"player": player, "team": hist[-1]["t"], "game": g["id"] if g else "", "time": g["time"] if g else "",
                        "stat": stat, "model": rnd(mean, 1), "main_line": pr["dk"] if pr else None,
                        "avg5": rnd(sum(vals[:5]) / len(vals[:5]), 1), "avg15": rnd(sum(vals) / len(vals), 1),
                        "last10": vals[:10][::-1], "rungs": ladder, "suggested": [r["line"] for r in pick],
                        "best_ev": best["ev"], "value_rungs": len(value)})
    out.sort(key=lambda x: (-x["value_rungs"], -x["best_ev"]))
    return out[:24]


# ------------------------------------------------------------------ permanent bet ledger
def _status(r):
    return "Pending" if not r else {"W": "Won", "L": "Lost", "P": "Push"}.get(r, r)


def ledger_rows(settled_bets, potd_hist, ladder):
    rows = []
    for b in settled_bets:
        rows.append({"id": f"model|{b['date']}|{b['game']}|{b['market']}|{b['side']}", "source": "Model bet",
                     "date": b["date"], "game": b["game"], "market": b["market"], "pick": b["side"],
                     "price": b.get("price"), "edge": b.get("edge"), "result": b.get("result"),
                     "status": _status(b.get("result")), "actual": b.get("actual"), "units": b.get("units"),
                     "why": f"Model edge of {b.get('edge')} vs the DraftKings line"})
    for h in potd_hist:
        rows.append({"id": f"potd|{h['date']}", "source": "Pick of the Day", "date": h["date"], "game": h["game"],
                     "market": h.get("type", "Prop"), "pick": h.get("label") or f"{h.get('player')} {h.get('side')} {h.get('line')} {h.get('stat')}",
                     "price": h.get("price"), "edge": h.get("edge"), "confidence": h.get("score"),
                     "result": h.get("result"), "status": _status(h.get("result")), "actual": h.get("actual"),
                     "units": h.get("units"), "why": " | ".join(h.get("reasons") or []),
                     "picked_at": h.get("picked_at"), "graded_at": h.get("graded_at"), "clv": h.get("clv"),
                     "open": [h.get("open_line"), h.get("open_price")], "close": [h.get("close_line"), h.get("close_price")]})
    for st in ladder.get("steps", []):
        r = st.get("result")
        rows.append({"id": f"ladder|{st['attempt']}|{st['day']}|{st['date']}", "source": "Ladder Challenge",
                     "date": st["date"], "game": st["game"], "market": st.get("type", "Prop"), "pick": st.get("label"),
                     "price": st.get("price"), "edge": st.get("edge"), "confidence": st.get("score"),
                     "attempt": st["attempt"], "day": st["day"], "stake": st.get("stake"), "to_win": st.get("to_win"),
                     "bankroll_after": st.get("bankroll_after"), "win_chance": st.get("p_win"),
                     "result": r, "status": _status(r), "actual": st.get("actual"),
                     "units": (rnd(st["stake"] * (decimal(st["price"]) - 1), 2) if r == "W" else -st["stake"] if r == "L" else 0.0) if r else None,
                     "why": " | ".join([st.get("why_ladder", "")] + (st.get("reasons") or [])).strip(" |"),
                     "picked_at": st.get("picked_at"), "graded_at": st.get("graded_at"), "clv": st.get("clv"),
                     "open": [st.get("open_line"), st.get("open_price")], "close": [st.get("close_line"), st.get("close_price")]})
    return rows


def update_ledger(settled_bets, potd_hist, ladder, now):
    """Upsert every bet into data/records/bets_YYYY-MM.json (never deletes) and rebuild summary.json."""
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    by_month = {}
    for r in ledger_rows(settled_bets, potd_hist, ladder):
        by_month.setdefault(r["date"][:7], []).append(r)
    for month, rows in by_month.items():
        path = RECORDS_DIR / f"bets_{month}.json"
        cur = {x["id"]: x for x in load(path, {"bets": []}).get("bets", [])}
        for r in rows:
            cur[r["id"]] = {**cur.get(r["id"], {}), **{k: v for k, v in r.items() if v is not None or k not in cur.get(r["id"], {})}}
        order = {"Pick of the Day": 0, "Ladder Challenge": 1, "Model bet": 2}
        bets = sorted(cur.values(), key=lambda x: (x["date"], -order.get(x["source"], 9)), reverse=True)
        save(path, {"month": month, "updated": now.strftime("%Y-%m-%d %I:%M %p ET"), "count": len(bets), "bets": bets}, indent=1)
    # summary across every month on disk
    summary, recent = {}, []
    for f in sorted(RECORDS_DIR.glob("bets_*.json"), reverse=True):
        data = load(f, {"bets": []})
        for b in data["bets"]:
            for key in (("All", b["source"]), (data.get("month", f.stem[5:]), b["source"])):
                t = summary.setdefault(key[0], {}).setdefault(key[1], {"bets": 0, "won": 0, "lost": 0, "push": 0, "pending": 0, "units": 0.0})
                t["bets"] += 1
                t[{"W": "won", "L": "lost", "P": "push"}.get(b.get("result"), "pending")] += 1
                if b["source"] != "Ladder Challenge":
                    t["units"] = rnd(t["units"] + (b.get("units") or 0), 2)
        if len(recent) < 600:
            recent += data["bets"][:600 - len(recent)]
    save(RECORDS_DIR / "summary.json", {"updated": now.strftime("%Y-%m-%d %I:%M %p ET"),
                                        "note": "Units are 1 unit per bet for Model bets and Pick of the Day. Ladder steps are tracked in dollars on the ladder itself.",
                                        "by_month": summary}, indent=1)
    return {"summary": summary, "bets": recent}


def settle(pending, plog, state):
    """Score frozen predictions for games that have finished. Updates state in place."""
    names = {}
    for eid, g in plog["games"].items():
        names[eid] = {norm(p["n"]): p for p in g["players"]}
    done = []
    for eid, pg in pending["games"].items():
        lg = plog["games"].get(eid)
        if not lg or pg.get("settled"):
            continue
        label = f'{pg["away"]} @ {pg["home"]}'
        margin = lg["home_pts"] - lg["away_pts"]
        total = lg["home_pts"] + lg["away_pts"]
        bets = []
        dk = pg.get("dk")
        if dk:
            hs = float(dk["spread"])
            sedge = hs - pg["model"]["spread"]            # >0: model likes home more than market
            if abs(sedge) >= BET_EDGE_GAME:
                home_pick = sedge > 0
                cover = margin + hs
                res = "P" if cover == 0 else ("W" if (cover > 0) == home_pick else "L")
                side = f'{pg["home"]} {dk["spread"]}' if home_pick else f'{pg["away"]} {sgn(-hs)}'
                bets.append(("Spread", side, rnd(sedge), res, f"{lg['away_pts']}-{lg['home_pts']}"))
            tedge = pg["model"]["total"] - dk["total"]
            if abs(tedge) >= BET_EDGE_GAME:
                over = tedge > 0
                res = "P" if total == dk["total"] else ("W" if (total > dk["total"]) == over else "L")
                bets.append(("Total", f'{"Over" if over else "Under"} {dk["total"]:g}', rnd(tedge), res, f"Total {total}"))
        for p in pending["props"].get(eid, []):
            row = names[eid].get(norm(p["player"]))
            if not row:
                continue
            actual = {"PTS": row["pts"], "REB": row["reb"], "AST": row["ast"], "3PM": row["tpm"],
                      "BLK": row["blk"], "STL": row["stl"], "PRA": row["pts"] + row["reb"] + row["ast"],
                      "PR": row["pts"] + row["reb"], "PA": row["pts"] + row["ast"],
                      "RA": row["reb"] + row["ast"]}.get(p["stat"])
            if actual is None:
                continue
            key = f'{p["player"]}|{p["stat"]}'
            old = state["residuals"]["player"].get(key, 0.0)
            state["residuals"]["player"][key] = rnd((1 - RES_BETA) * old + RES_BETA * (actual - p["model"]), 2)
            edge = p["model"] - p["dk"]
            if abs(edge) >= BET_EDGE_PROP:
                over = edge > 0
                res = "P" if actual == p["dk"] else ("W" if (actual > p["dk"]) == over else "L")
                price = p.get("over") if over else p.get("under")
                bets.append(("Player Prop", f'{p["player"]} {"Over" if over else "Under"} {p["dk"]:g} {p["stat"]}',
                             rnd(edge), res, f'{actual:g} {p["stat"]}', price))
        if dk:
            state["accuracy"].insert(0, {"date": lg["date"], "game": label, "model_spread": pg["model"]["spread"],
                                         "dk_spread": float(dk["spread"]), "actual_margin": margin,
                                         "model_total": pg["model"]["total"], "dk_total": dk["total"], "actual_total": total})
        for bet in bets:
            market, side, edge, res, actual = bet[:5]
            price = bet[5] if len(bet) > 5 else None
            win = (price / 100 if price and price > 0 else 100 / -price if price else WIN_UNITS)
            units = rnd(win, 2) if res == "W" else (-1.0 if res == "L" else 0.0)
            state["settled_bets"].insert(0, {"date": lg["date"], "game": label, "market": market, "side": side,
                                             "edge": edge, "result": res, "units": rnd(units, 2), "actual": actual})
            rec = state["record"]
            rec["wins"] += res == "W"
            rec["losses"] += res == "L"
            rec["units"] = rnd(rec["units"] + units, 2)
        state["residuals"]["games_learned"] += 1
        pg["settled"] = True
        done.append(eid)
    state["settled_bets"] = state["settled_bets"][:500]
    state["accuracy"] = state["accuracy"][:1000]
    log(f"settled {len(done)} games")


# -------------------------------------------------------------------- main
def main():
    now = datetime.now(ET)
    global CREDITS
    CREDITS = Credits(now)
    slot = "overnight" if now.hour < 6 else ("pretip" if now.hour >= 17 else "morning")
    log(f"run slot: {slot}; daily credit limit {DAILY_CREDIT_LIMIT}, used so far today {CREDITS.state['used_today']}")
    today = now.date()
    prev = load(OUT, {})
    plog = load(LOG_PATH, {"games": {}, "through": None})
    if plog.get("v") != 2:                       # older log lacks shot/turnover detail: rebuild once
        plog = {"games": {}, "through": None, "v": 2}
    pending = load(PENDING_PATH, {"games": {}, "props": {}})

    state = {
        "residuals": {"player": prev.get("residuals", {}).get("player") or {}, "team": [],
                      "games_learned": prev.get("residuals", {}).get("games_learned", 0),
                      "history": [], "alpha": 0.35},
        "settled_bets": prev.get("settled_bets", []),
        "record": prev.get("record") or {"wins": 0, "losses": 0, "units": 0},
        "prediction_history": prev.get("prediction_history", []),
        "accuracy": prev.get("accuracy", []),
    }
    if isinstance(state["residuals"]["player"], list):
        state["residuals"]["player"] = {}

    # 1. results in
    ingest(plog, today)
    # 2. score what we predicted
    settle(pending, plog, state)

    # 3. slate: yesterday..tomorrow
    slate = []
    for off in (-1, 0, 1):
        evs = scoreboard(today + timedelta(days=off))
        if evs is None:
            if off == 0:
                log("could not load today's scoreboard; leaving existing data untouched")
                return 1
            continue
        slate += [parse_event(e) for e in evs]
    t_iso = today.isoformat()
    # Only current games are published: live, upcoming from today on, and today's finals.
    # Older results live in data/ (for settlement), never on the page.
    slate = [g for g in slate if g["status"] == "Live"
             or (g["status"] == "Final" and g["date"] == t_iso)
             or (g["status"] == "Upcoming" and g["date"] >= t_iso)]
    if not slate:
        log("no games today or tomorrow; publishing an empty slate")

    ratings, lg = team_ratings(plog)
    positions = refresh_positions(today)
    dfn = team_defense(plog, positions)
    idx = player_index(plog)
    injuries = fetch_injuries(slate)
    if injuries is None:
        injuries = prev.get("injuries", [])
    boosts = injury_boosts(idx, injuries, positions)
    out_set = {(norm(i["player"])) for i in injuries if i["status"].lower() == "out"}
    upcoming = [g for g in slate if g["status"] == "Upcoming"]
    # Spend in priority order: game lines (3) -> props for today's games, earliest tip first -> futures (once a day).
    # Overnight runs skip props: most lines are not posted yet and they move before tip-off.
    gml = odds_game_lines() if upcoming else {}
    def tip(g):
        m = re.match(r"(\d+):(\d+) (AM|PM)", g["time"])
        return (int(m.group(1)) % 12 + (12 if m.group(3) == "PM" else 0)) * 60 + int(m.group(2)) if m else 9999
    todays_up = sorted([g for g in upcoming if g["date"] == t_iso], key=tip)
    prop_lines = odds_props(todays_up, idx) if (todays_up and slot != "overnight") else {}
    for g in upcoming:
        hit = gml.get((NAME_FIX.get(g["names"][g["away"]], g["names"][g["away"]]), NAME_FIX.get(g["names"][g["home"]], g["names"][g["home"]])))
        if hit:
            g["_lines"] = {**hit, "provider": "DraftKings"}
    rest = team_rest(plog, slate)
    standings = fetch_standings() or prev.get("standings")
    fut_fresh = (prev.get("futures") or {}).get("updated", "")[:5] == now.strftime("%m/%d")
    futures = (None if fut_fresh else fetch_futures()) or prev.get("futures")

    games, props = [], []
    for g in slate:
        eid = g["event_id"]
        frozen = pending["games"].get(eid)
        if frozen and g["status"] != "Upcoming":
            model, dk = frozen["model"], frozen.get("dk")        # lines frozen at tip-off
        else:
            model, dk = model_game(g, ratings, lg), g["_lines"]
            if dk:
                dk = {k: v for k, v in dk.items() if k != "provider"}
            pending["games"][eid] = {"away": g["away"], "home": g["home"], "date": g["date"],
                                     "model": model, "dk": dk}
        row = {"id": g["id"], "date": g["date"], "time": g["time"], "away": g["away"], "home": g["home"],
               "model": model, "dk": dk or {"spread": "N/A", "total": 0, "ml_away": "N/A", "ml_home": "N/A"},
               "status": g["status"], "rest": rest.get(eid, {})}
        if dk:
            row["spread_edge"] = rnd(float(dk["spread"]) - model["spread"])
            row["total_edge"] = rnd(model["total"] - dk["total"])
            side = "home" if row["spread_edge"] > 0 else "away"
            row["note"] = (f'Model {g["home"]} {sgn(model["spread"])}, total {model["total"]:g}; '
                           f'leans {g[side]} vs the market spread.' if abs(row["spread_edge"]) >= 0.5
                           else "Model and market are close; no strong spread lean.")
        else:
            row["spread_edge"], row["total_edge"] = 0, 0
            row["note"] = "No market line available yet."
        if "score" in g:
            row["score"] = g["score"]
            if g["status"] == "Final":
                row["actual_margin"] = g["score"]["home"] - g["score"]["away"]
                row["actual_total"] = g["score"]["home"] + g["score"]["away"]
        games.append(row)

        # props
        if g["status"] == "Upcoming" and eid in prop_lines:
            fresh = []
            for player, stat, line, over_px, under_px in prop_lines[eid]:
                hist = idx.get(norm(player))
                if not hist or norm(player) in out_set:
                    continue
                team = hist[-1]["t"]
                if team not in (g["home"], g["away"]):
                    continue
                opp = g["away"] if team == g["home"] else g["home"]
                bst = boosts.get(norm(player))
                proj = project_player(hist, dfn, opp, bst, sched_ctx(row, team))
                if not proj or stat not in proj:
                    continue
                fresh.append({"player": hist[-1]["n"], "team": team, "opp": opp, "game": g["id"], "stat": stat,
                              "model": proj[stat], "matchup": proj["_matchup"][stat],
                              "injury": proj["_injury"][stat], "inj_out": sorted(set(bst["out"])) if bst else [],
                              "sched": proj["_sched"], "sched_notes": proj["_sched_notes"],
                              "dk": line, "over": over_px, "under": under_px})
            if fresh:
                pending["props"][eid] = fresh
        elif g["status"] == "Upcoming" and pending["props"].get(eid):
            # No new lines this run (saves credits): keep the saved DraftKings lines but re-project them
            # with the latest box scores, matchup and injury data. Costs nothing.
            pending["props"][eid] = [p for p in pending["props"][eid] if norm(p["player"]) not in out_set]
            for p in pending["props"][eid]:
                hist = idx.get(norm(p["player"]))
                if not hist:
                    continue
                opp = g["away"] if p["team"] == g["home"] else g["home"]
                bst = boosts.get(norm(p["player"]))
                proj = project_player(hist, dfn, opp, bst, sched_ctx(row, p["team"]))
                if proj and p["stat"] in proj:
                    p.update(opp=opp, model=proj[p["stat"]], matchup=proj["_matchup"][p["stat"]],
                             injury=proj["_injury"][p["stat"]], inj_out=sorted(set(bst["out"])) if bst else [],
                             sched=proj["_sched"], sched_notes=proj["_sched_notes"])
        for p in pending["props"].get(eid, []):
            e = rnd(p["model"] - p["dk"])
            props.append({**p, "edge": e, "side": "Over" if e > 0 else "Under"})

    add_tier_hits(props, idx, dfn)

    # 3b. Pick of the Day: re-picked each run until its game tips off, then locked and saved for good
    potd_log = load(POTD_LOG, None)
    potd_hist = potd_log["picks"] if potd_log else prev.get("potd_history", [])
    grade_potd(potd_hist, plog)
    todays = next((h for h in potd_hist if h["date"] == t_iso), None)
    locked = bool(todays) and bool(todays.get("result") or next((g["status"] != "Upcoming" for g in slate if g["event_id"] == todays["eid"]), True))
    potd_alts = prev.get("potd_alts", []) if locked else []
    if todays and locked and not todays.get("locked_at"):
        todays["locked_at"] = now.strftime("%Y-%m-%d %I:%M %p ET")
    pick, new_alts, all_cands = pick_of_the_day(props, games, slate, idx, plog, injuries, t_iso)
    add_win_probs(all_cands)
    if not locked:
        potd_alts = new_alts
        potd_hist = [h for h in potd_hist if h["date"] != t_iso]
        if pick:
            pick["picked_at"] = now.strftime("%Y-%m-%d %I:%M %p ET")
            potd_hist.insert(0, carry_open(pick, todays))
    # 3c. Ladder challenge (uses the same qualifying bets, but picks the safest one with a real payout)
    ladder = run_ladder(load(LADDER_PATH, {}), plog, all_cands, slate, t_iso, now)
    track_clv(potd_hist[:5] + ladder["steps"][:5], games, props, slate, now)

    # 3d. Alt-line ladders: lowest spending priority, morning run only, for the games of the best props
    alt_ladders = prev.get("alt_ladders", []) if prev.get("alt_date") == t_iso else []
    alt_date = prev.get("alt_date") if alt_ladders else None
    if slot == "morning" and todays_up:
        ranked = []
        for c in sorted(all_cands, key=lambda c: -c["score"]):
            gg = next((g for g in todays_up if g["event_id"] == c["eid"]), None)
            if gg and gg not in ranked:
                ranked.append(gg)
        for g in todays_up:
            if g not in ranked:
                ranked.append(g)
        alt = odds_alt_lines(ranked[:ALT_GAMES], slate)
        built = build_alt_ladders(alt, slate, idx, props, injuries)
        if built:
            alt_ladders, alt_date = built, t_iso

    log("potd: " + (potd_hist[0]["label"] + f" ({potd_hist[0]['type']})" if potd_hist and potd_hist[0]["date"] == t_iso else "none today"))

    # 4. fading learning rate
    gl = state["residuals"]["games_learned"]
    state["residuals"]["alpha"] = rnd(0.12 + 0.33 * math.exp(-gl / 60), 3)

    # 5. history snapshot (same shape the page writes)
    stamp = now.strftime("%Y-%m-%dT%H:%M")
    top = sorted([p for p in props if abs(p["edge"]) >= 0.5], key=lambda p: -abs(p["edge"]))[:30]
    state["prediction_history"].append({
        "timestamp": stamp,
        "games": [{"id": g["id"], "matchup": f'{g["away"]} @ {g["home"]}', "model_spread": g["model"]["spread"],
                   "model_total": g["model"]["total"], "dk_spread": g["dk"]["spread"], "dk_total": g["dk"]["total"]}
                  for g in games if g["status"] != "Final"],
        "props": [{"player": p["player"], "stat": p["stat"], "model": p["model"], "dk": p["dk"], "edge": p["edge"]}
                  for p in top]})
    state["prediction_history"] = state["prediction_history"][-60:]

    # prune stale pending
    cutoff = (today - timedelta(days=10)).isoformat()
    for eid in [e for e, v in pending["games"].items() if v["date"] < cutoff]:
        pending["games"].pop(eid, None)
        pending["props"].pop(eid, None)

    note = ("Model data updates automatically each day." if ODDS_KEY else
            "Player props are off until an ODDS_API_KEY is added; game lines come from ESPN.")
    data = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "season": SEASON,
            "games": games, "props": props, "teams": TEAM_NAMES, "injuries": injuries, "injuries_updated": now.strftime("%m/%d %I:%M %p ET"),
            "residuals": state["residuals"],
            "model_notes": ["Base: projected minutes x recent per-minute rates (last 10 games, decay-weighted)",
                            "Adjustments: props scaled to the opponent's defense in that stat against the player's position (G/F/C, capped +/-10%) and the expected game pace (+/-5%), after removing the effect of defenses already faced; teammate injury boost hands an OUT player's minutes and usage to healthy teammates (capped +20%); small game-script and rest nudges (blowout risk past an 8-pt spread, back-to-backs, extra rest; capped +/-5% combined); home court for games; injured players skipped",
                            f"Residual layer: alpha fades from 0.45 toward 0.12 as games are learned (now {state['residuals']['alpha']})",
                            "Benchmark: DraftKings via The Odds API (games and props), ESPN as backup",
                            "Bets are scored at 1 unit risked at the posted DraftKings price (-110 when no price was posted)"],
            "prediction_history": state["prediction_history"], "notes": note,
            "settled_bets": state["settled_bets"], "record": state["record"],
            "accuracy": state["accuracy"], "standings": standings,
            "potd_history": potd_hist[:400], "potd_alts": potd_alts,
            "ladder": {**{k: v for k, v in ladder.items() if k != "steps"}, "steps": ladder["steps"][:120]},
            "alt_ladders": alt_ladders, "alt_date": alt_date,
            "clv": {"potd": clv_summary(potd_hist), "ladder": clv_summary(ladder["steps"]),
                    "all": clv_summary(potd_hist + ladder["steps"])},
            "credits": {"limit": DAILY_CREDIT_LIMIT, "used_today": CREDITS.state["used_today"], "remaining": CREDITS.state.get("remaining"),
                        "allowed_now": CREDITS.allowance(), "skipped": CREDITS.skipped, "slot": slot,
                        "calls": CREDITS.state["calls"][-20:], "history": CREDITS.state.get("history", [])[:14],
                        "days_to_reset": CREDITS.days_to_reset(), "month_used": CREDITS.month_used(),
                        "month_limit": MONTHLY_CREDIT_LIMIT},
            "futures": {"title": futures, "book": futures[0].get("book", ""), "updated": now.strftime("%m/%d %I:%M %p ET")} if isinstance(futures, list) else futures}

    # validate before replacing anything
    json.dumps(data)
    assert isinstance(data["games"], list) and isinstance(data["props"], list)
    save(LOG_PATH, plog)
    save(PENDING_PATH, pending)
    CREDITS.save()
    ladder["steps"] = ladder["steps"][:1000]
    save(LADDER_PATH, ladder, indent=1)
    records = update_ledger(state["settled_bets"], potd_hist, ladder, now)
    data["records"] = records
    save(POTD_LOG, {"note": "Every Pick of the Day with its reasons and result. Never trimmed.", "picks": potd_hist}, indent=1)
    save(OUT, data, indent=1)
    save(ROOT / "player_games.json", build_game_logs(idx, slate, now.strftime("%m/%d %I:%M %p ET")))
    log(f"wrote {len(games)} games, {len(props)} props, {len(injuries)} injuries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
