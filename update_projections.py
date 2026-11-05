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
ALT_LOG = ROOT / "data" / "alt_log.json"         # every suggested alt-line ladder, graded after the game
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
# Free tip-off checks (7-10 PM ET): no Odds API calls at all. Injuries and box scores come from ESPN (free) and
# the lines saved at the last paid run are reused, so a pick whose player is ruled out can be swapped for free.
TIP_CHECK = os.environ.get("NO_PAID_CALLS", "") == "1"
MAX_PROP_EVENTS = int(os.environ.get("MAX_PROP_EVENTS", "30"))
DAILY_CREDIT_LIMIT = int(os.environ.get("DAILY_CREDIT_LIMIT", "250") or 250)   # hard cap per ET day (unused credits roll forward)
MONTHLY_CREDIT_LIMIT = int(os.environ.get("MONTHLY_CREDIT_LIMIT", "5500") or 5500)  # this site's share of the plan
# Up to 10 sportsbooks bill as ONE region on The Odds API, so these cost the same as DraftKings alone.
# DraftKings stays the line the model is compared and graded against; the others supply the best price.
BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "fanatics", "betrivers", "hardrockbet", "ballybet", "betparx"]
BOOK_NAMES = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars", "espnbet": "ESPN BET",
              "fanatics": "Fanatics", "betrivers": "BetRivers", "hardrockbet": "Hard Rock", "ballybet": "Bally Bet", "betparx": "betPARX"}
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
BET_EDGE_PROP = 0.8  # min edge to count a prop bet (points), plus the share-of-line rules below
BET_EDGE_PCT_OVER = 0.08   # Over: edge must be 8%+ of the line (0.8 on a 30.5 PRA line is noise)
BET_EDGE_PCT_UNDER = 0.12  # Under: 12%+, since small-edge Unders have lost the most
BET_EDGE_PCT_MAX = 0.35    # an edge past 35% of the line usually means the model is missing news: skip it
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
        if not ODDS_KEY or TIP_CHECK:
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
MATCHUP_CAP = 0.12   # combined matchup cap (tonight's opponent and the defenses already faced, together)
POS_SHRINK_K = 20    # position-level defense is noisier than team-level, so it is pulled harder toward average
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
            shr = POS_SHRINK_K * L["min"] / n                    # pseudo-minutes: K games of a typical team
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
        # the whole matchup move (tonight vs the defenses already faced) stays within MATCHUP_CAP
        net = {k: _clamp(mx[k] / (faced[k] or 1), MATCHUP_CAP) for k in DEF_STATS}
        mx = {k: net[k] for k in DEF_STATS}
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
                          bookmakers=",".join(BOOKS), markets=",".join(PROP_MARKETS), oddsFormat="american")
        found, others = {}, []
        for bk in (j or {}).get("bookmakers", []):
            for m in bk.get("markets", []):
                stat = PROP_MARKETS.get(m["key"])
                if not stat:
                    continue
                for o in m.get("outcomes", []):
                    if o.get("point") is None or o.get("name") not in ("Over", "Under"):
                        continue
                    if bk.get("key") != "draftkings":
                        others.append((bk.get("key"), (o.get("description", ""), stat), float(o["point"]), o["name"], o.get("price")))
                        continue
                    key = (o.get("description", ""), stat)
                    row = found.setdefault(key, {"line": float(o["point"]), "over": None, "under": None})
                    if float(o["point"]) != row["line"]:
                        continue                      # keep the main (first) line only
                    row["over" if o["name"] == "Over" else "under"] = o.get("price")
        # best price across books, at DraftKings' line only (a different line is a different bet)
        for book, key, pt, side, px in others:
            row = found.get(key)
            if not row or pt != row["line"] or px is None:
                continue
            k = "best_over" if side == "Over" else "best_under"
            cur = row.get(k) or ([row["over" if side == "Over" else "under"], "draftkings"] if row["over" if side == "Over" else "under"] is not None else None)
            if cur is None or px > cur[0]:
                row[k] = [px, book]
        out[g["event_id"]] = [(pl, st, r["line"], r["over"], r["under"],
                               {k: r[k] for k in ("best_over", "best_under") if r.get(k) and r[k][1] != "draftkings"})
                              for (pl, st), r in found.items() if r["over"] is not None or r["under"] is not None]
    return out


def odds_game_lines():
    """DraftKings spread/total/moneyline for every NBA game in one call. {(away_name, home_name): lines}"""
    if not ODDS_KEY:
        return {}
    j = CREDITS.spend(f"{ODDS}/odds", 3, "game lines", bookmakers="draftkings",
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
        if e.get("result"):
            if e.get("clv") and not e["clv"].get("final"):
                e["clv"]["final"] = True              # graded: the last line we saw before tip is the close
            continue
        if not e.get("price"):
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


def with_residual(state, player, stat, base):
    """Learning loop: nudge a projection by the player's recent miss on that stat (alpha fades as games are
    learned), capped at +/-15% so it can never swing a line drastically. Done here, once, so the page, the picks
    and the grading all use the same number."""
    res = state["residuals"]["player"].get(f"{player}|{stat}", 0.0)
    a = state["residuals"].get("alpha", 0.35)
    if not res or abs(res) <= 0.1:
        return base
    return rnd(base + max(-0.15 * base, min(0.15 * base, a * res)))


def p_over_for(stat, mean, line, hist):
    """Model chance of going over, same distribution the picks use (saved so calibration can grade it)."""
    if stat not in STAT_VAL:
        return None
    vals = [STAT_VAL[stat](r) for r in hist if r["min"] >= 5][-15:]
    return rnd(prob_over(stat, mean, line, vals), 3)


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


def best_price_note(p, over, price):
    px, book = p["best_over" if over else "best_under"]
    if px is None or price is None or px <= price:
        return None
    return f"Best price: {fmt_px(px)} at {BOOK_NAMES.get(book, book)} (DraftKings {fmt_px(price)}; the record is graded at DraftKings)"


def pick_of_the_day(props, games, slate, idx, plog, injuries, today_iso, paused=None):
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
        if norm(p["player"]) in hurt or p["stat"] not in STAT_VAL or p["stat"] in (paused or {}):
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
                                  *[x for x in [best_price_note(p, over, price) if p.get("best_over" if over else "best_under") else None] if x],
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
        if 1.5 <= abs(se) <= 6 and "Spread" not in (paused or {}) and dk.get("spread_px_home" if se > 0 else "spread_px_away"):
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
        if 3 <= abs(te) <= 12 and "Total" not in (paused or {}) and dk.get("over_px" if te > 0 else "under_px"):
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
            if px is None or not (-250 <= px <= 250) or "Moneyline" in (paused or {}):
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


# ------------------------------------------------------------------ calibration (honest win chances)
# The model's raw win chances run hot: picks it rated 80% have hit far less than that. Every win chance used to
# rank or show a pick goes through this map, built from every graded model lean in data/calibration.json.
# Thin buckets lean on the overall pattern (how much of the claimed edge has really shown up), and the map can
# only go up as the raw chance goes up. It updates itself after every run as more leans are graded.
CAL_K = 30
CAL_MAP = []
CAL_BLEND_MIN = 300   # graded leans on the blended win chance needed before the map switches to them


def blend_record(cal, p, res):
    """Grade the blended win chance (model + recent hit rate, the number picks are ranked on) so the
    calibration map can be built from the exact number it is applied to."""
    if res == "P" or p is None:
        return
    p = max(0.5, min(0.999, p))
    b = str(min(90, int(p * 100) // 5 * 5))
    r = cal.setdefault("blend", {}).setdefault(b, {"n": 0, "w": 0, "p": 0.0})
    r["n"] += 1; r["w"] += res == "W"; r["p"] = rnd(r["p"] + p, 3)


def build_cal_map(cal, market="ALL"):
    rows = [r for _, r in sorted(((cal or {}).get("buckets", {}).get(market) or {}).items(), key=lambda kv: int(kv[0])) if r.get("n", 0) > 0]
    if not rows:
        return []
    # share of the model's claimed edge (over 50%) that has actually shown up, across every lean
    n, w, pp = sum(r["n"] for r in rows), sum(r["w"] for r in rows), sum(r["p"] for r in rows)
    keep = max(0.0, min(1.0, (w - 0.5 * n) / (pp - 0.5 * n))) if pp - 0.5 * n > 1 else 1.0
    pts = []
    for r in rows:
        pavg = r["p"] / r["n"]
        prior = 0.5 + (pavg - 0.5) * keep         # thin buckets lean on the overall pattern, not the raw claim
        pts.append([pavg, (r["w"] + CAL_K * prior) / (r["n"] + CAL_K), r["n"] + CAL_K])
    merged = []                                   # pool adjacent violators: higher raw chance never maps lower
    for x, y, w in pts:
        merged.append([x * w, y * w, w])
        while len(merged) > 1 and merged[-2][1] / merged[-2][2] > merged[-1][1] / merged[-1][2]:
            a = merged.pop()
            merged[-1] = [merged[-1][0] + a[0], merged[-1][1] + a[1], merged[-1][2] + a[2]]
    return [(x / w, y / w) for x, y, w in merged]


def calibrate(p, cmap=None):
    """Raw chance (0-1) -> calibrated chance. Leans under 50% are mirrored."""
    cmap = CAL_MAP if cmap is None else cmap
    if not cmap or p is None:
        return p
    flip = p < 0.5
    q = 1 - p if flip else p
    if q <= cmap[0][0]:
        x0, y0 = cmap[0]
        y = 0.5 + (q - 0.5) * (y0 - 0.5) / (x0 - 0.5) if x0 > 0.5 else y0
    elif q >= cmap[-1][0]:
        y = cmap[-1][1]
    else:
        for (x0, y0), (x1, y1) in zip(cmap, cmap[1:]):
            if x0 <= q <= x1:
                y = y0 + (y1 - y0) * (q - x0) / (x1 - x0) if x1 > x0 else y1
                break
    y = max(0.3, min(0.97, y))
    return 1 - y if flip else y


def add_win_probs(cands):
    """Attach p_win: 60% model distribution + 40% recent hit rate (shrinks small samples toward the model), then
    calibrated against how the model's leans have actually done (p_win_raw keeps the uncalibrated number)."""
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
        raw = 0.6 * pm + 0.4 * hr
        c["p_win_raw"] = rnd(raw * 100, 1)
        c["p_win"] = rnd(calibrate(raw) * 100, 1)
        dec = 1 + (c["price"] / 100 if c["price"] > 0 else 100 / -c["price"])
        c["ev"] = rnd((c["p_win"] / 100 * dec - 1) * 100, 1)


def decimal(price):
    return 1 + (price / 100 if price > 0 else 100 / -price)


# ------------------------------------------------------------------ ladder challenge
def bet_key(b):
    """What makes two picks 'the same play': the same player (any stat), or the same game bet type."""
    if not b:
        return None
    return ("P", norm(b["player"])) if b.get("player") else ("G", b.get("game"), b.get("type"))


def step_keys(step):
    """bet_key of every bet in a ladder step (each leg of a parlay step)."""
    if not step:
        return set()
    return {bet_key(l) for l in step.get("legs") or [step]} - {None}


def step_eids(step):
    return [l["eid"] for l in step.get("legs") or [step] if l.get("eid")]


def american(dec):
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


# Multi-pick ladder steps: allowed, never required. A 2-3 leg parlay (each leg from a different game) replaces the
# single bet only when it gives a better chance of finishing the ladder. Model win chances run hot (see calibration),
# so both options are compared on win chances shrunk toward 50%, which costs parlays more than singles.
PARLAY_SHRINK = 1.0          # win chances are calibrated now (see calibrate), so no extra shrink
PARLAY_MIN_WIN = 0.45        # calibrated chance the whole parlay hits
PARLAY_LEG_MIN = 60          # every leg's own calibrated win chance (%)
PARLAY_MARGIN = 1.05         # must beat the single by 5% to be worth the extra legs


def _shrunk(p_pct):
    return 0.5 + (p_pct / 100 - 0.5) * PARLAY_SHRINK


def ladder_efficiency(p, dec):
    """ln(payout) / -ln(win chance): higher means fewer, safer steps to the goal (better odds of finishing)."""
    return math.log(dec) / -math.log(p) if 0 < p < 1 and dec > 1 else 0.0


def ladder_parlay(cands, avoid, single):
    from itertools import combinations
    best_leg = {}
    for c in cands:
        if c.get("ev", -1) <= 0 or c.get("p_win", 0) < PARLAY_LEG_MIN or not (-300 <= c["price"] <= 120) or bet_key(c) in avoid:
            continue
        cur = best_leg.get(c["eid"])
        if cur is None or (c["p_win"], c["ev"]) > (cur["p_win"], cur["ev"]):
            best_leg[c["eid"]] = c                       # one leg per game, so the legs don't move together
    pool = sorted(best_leg.values(), key=lambda c: (-c["p_win"], -c["ev"]))[:8]
    best, best_eff = None, 0.0
    for n in (2, 3):
        for legs in combinations(pool, n):
            p = math.prod(_shrunk(l["p_win"]) for l in legs)
            dec = math.prod(decimal(l["price"]) for l in legs)
            if p < PARLAY_MIN_WIN or p * dec <= 1:
                continue
            e = ladder_efficiency(p, dec)
            if e > best_eff:
                best, best_eff = legs, e
    if not best:
        return None
    single_eff = ladder_efficiency(_shrunk(single["p_win"]), decimal(single["price"])) if single else 0.0
    if best_eff <= single_eff * PARLAY_MARGIN:
        return None
    return best, best_eff, single_eff


LEG_KEYS = ("type", "label", "game", "eid", "time", "home", "away", "side", "line", "price", "model", "edge", "edge_pct",
            "hits", "chart", "reasons", "p_win", "p_model", "ev", "score", "player", "team", "stat")


def grade_parlay(st, plog):
    """A parlay step wins only if every leg wins. A pushed or voided leg drops out and the payout shrinks."""
    grade_potd(st["legs"], plog)
    res = [l.get("result") for l in st["legs"]]
    if "L" in res:
        st["result"] = "L"
    elif all(res):
        live = [l for l in st["legs"] if l["result"] == "W"]
        if live:
            st["result"] = "W"
            st["price"] = american(math.prod(decimal(l["price"]) for l in live))
        else:
            st["result"] = "P"
    else:
        return
    st["actual"] = " · ".join(f"{l['label']}: {l.get('actual') or ''} {'✓' if l.get('result') == 'W' else '✗' if l.get('result') == 'L' else '(push)' if l.get('result') == 'P' else '…'}".strip() for l in st["legs"])
    st["graded_at"] = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p ET")


def ladder_pick(cands, avoid=(), avoid_games=()):
    """Safest bet with a real payout: price -200 to +120, positive expected value. Highest win chance wins.
    Never the same play as today's Pick of the Day (avoid), and from a different game than the Pick of the Day
    whenever one qualifies (avoid_games), so one bad game can't sink both. Same game only as a fallback."""
    pool = [c for c in cands if -200 <= c["price"] <= 120 and c.get("ev", -1) > 0 and bet_key(c) not in avoid]
    pool.sort(key=lambda c: (-c["p_win"], -c["ev"]))
    other = [c for c in pool if c.get("eid") not in avoid_games]
    return (other or pool or [None])[0]


def run_ladder(lad, plog, cands, slate, t_iso, now, avoid=(), avoid_games=()):
    lad.setdefault("start", LADDER_START); lad.setdefault("goal", LADDER_GOAL)
    lad.setdefault("attempt", 1); lad.setdefault("bankroll", lad["start"]); lad.setdefault("steps", [])
    lad.setdefault("attempts", []); lad.setdefault("attempt_started", t_iso)
    stamp = now.strftime("%Y-%m-%d %I:%M %p ET")
    # 1. grade finished steps, oldest first, and move the bankroll
    grade_potd([x for x in lad["steps"] if x.get("type") != "Parlay"], plog)
    for x in lad["steps"]:
        if x.get("type") == "Parlay" and not x.get("result"):
            grade_parlay(x, plog)
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
    status = {g["event_id"]: g["status"] for g in slate}
    locked = bool(today) and bool(today.get("result") or any(status.get(e, "Final") != "Upcoming" for e in step_eids(today)))
    if today and locked:
        today.setdefault("locked_at", stamp)
        return lad
    lad["steps"] = [x for x in lad["steps"] if x["date"] != t_iso]
    pick = ladder_pick(cands, avoid, avoid_games)
    prev_step = today
    combo = ladder_parlay([c for c in cands if c.get("eid") not in avoid_games], avoid, pick)
    if combo:
        legs, eff, eff1 = combo
        legs = sorted(legs, key=lambda l: next((i for i, g in enumerate(slate) if g["event_id"] == l["eid"]), 99))
        dec = math.prod(decimal(l["price"]) for l in legs)
        p_raw = math.prod(l["p_win"] / 100 for l in legs)
        pick = {"type": "Parlay", "label": " + ".join(l["label"] for l in legs), "legs": [{k: l[k] for k in LEG_KEYS if k in l} for l in legs],
                "game": " / ".join(l["game"] for l in legs), "eid": legs[0]["eid"], "time": legs[0]["time"],
                "home": legs[0]["home"], "away": legs[0]["away"], "side": None, "line": None, "price": american(dec),
                "p_win": rnd(p_raw * 100, 1), "p_model": rnd(math.prod(l.get("p_model", l["p_win"]) / 100 for l in legs) * 100, 1),
                "ev": rnd((p_raw * dec - 1) * 100, 1), "score": rnd(min(l.get("score", 0) for l in legs), 1),
                "reasons": [f"Leg {i + 1}: {l['label']} ({fmt_px(l['price'])}): {l['p_win']:g}% to win" + (f". {l['reasons'][0]}" if l.get("reasons") else "")
                            for i, l in enumerate(legs)],
                "why_parlay": f"{len(legs)} picks from different games beat the best single bet for finishing the ladder "
                              f"({eff:.2f} vs {eff1:.2f} on shrunk win chances)"}
    if pick:
        if not lad.get("attempt_started"):
            lad["attempt_started"] = t_iso
        day = 1 + sum(1 for x in lad["steps"] if x["attempt"] == lad["attempt"])
        stake = lad["bankroll"]
        step = {k: pick[k] for k in ("type", "label", "game", "eid", "time", "home", "away", "side", "line", "price",
                                     "model", "edge", "edge_pct", "hits", "chart", "reasons", "p_win", "p_model", "ev", "score", "legs")
                if k in pick}
        for k in ("player", "team", "stat"):
            if k in pick:
                step[k] = pick[k]
        step.update({"date": t_iso, "attempt": lad["attempt"], "day": day, "stake": stake,
                     "to_win": rnd(stake * (decimal(pick["price"]) - 1), 2), "picked_at": stamp,
                     "why_ladder": (f"Multi-pick step. {pick['why_parlay']}: {pick['p_win']:g}% to win, {pick['ev']:+g}% expected value" if pick.get("legs") else
                                    f"Highest win chance on the board at a ladder-friendly price: {pick['p_win']:g}% to win, {pick['ev']:+g}% expected value")})
        lad["steps"].insert(0, carry_open(step, prev_step))
        log(f"ladder: attempt {lad['attempt']} day {day}, ${stake:g} on {pick['label']} ({fmt_px(pick['price'])})")
    else:
        log("ladder: no bet cleared the ladder filters today (rest day)")
    return lad


ALERTS_PATH = ROOT / "data" / "alerts.json"
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
SITE_URL = os.environ.get("SITE_URL", "https://joshgmiller2003.github.io/pjmodel/")

BADGES = [  # id, group, name, description, test(stats) -> bool, progress(stats) -> str
    ("first_win", "Pick of the Day", "First Win", "Win a Pick of the Day", lambda s: s["potd_w"] >= 1, lambda s: f"{min(s['potd_w'], 1)}/1"),
    ("heater", "Pick of the Day", "Heater", "3 Picks of the Day in a row", lambda s: s["potd_best"] >= 3, lambda s: f"{min(s['potd_best'], 3)}/3"),
    ("on_fire", "Pick of the Day", "On Fire", "5 Picks of the Day in a row", lambda s: s["potd_best"] >= 5, lambda s: f"{min(s['potd_best'], 5)}/5"),
    ("unstoppable", "Pick of the Day", "Unstoppable", "10 Picks of the Day in a row", lambda s: s["potd_best"] >= 10, lambda s: f"{min(s['potd_best'], 10)}/10"),
    ("ten_wins", "Pick of the Day", "Double Digits", "10 Pick of the Day wins", lambda s: s["potd_w"] >= 10, lambda s: f"{min(s['potd_w'], 10)}/10"),
    ("winning_month", "Pick of the Day", "Winning Month", "A month with 10+ graded picks at 55%+", lambda s: s["win_month"], lambda s: "best so far " + s["best_month"] if s["best_month"] != "-" else "no graded month yet"),
    ("climber", "Ladder", "Climber", "Reach Day 3 of a ladder", lambda s: s["lad_best_day"] >= 3, lambda s: f"{min(s['lad_best_day'], 3)}/3"),
    ("day5", "Ladder", "Ladder Day 5", "5 ladder wins in a row", lambda s: s["lad_best_day"] >= 5, lambda s: f"{min(s['lad_best_day'], 5)}/5"),
    ("doubled", "Ladder", "Doubled Up", "Grow a ladder to 2x the start", lambda s: s["lad_peak"] >= 2 * s["lad_start"], lambda s: f"${s['lad_peak']:g} / ${2 * s['lad_start']:g}"),
    ("triple", "Ladder", "Triple Digits", "Grow a ladder to $100", lambda s: s["lad_peak"] >= 100, lambda s: f"${s['lad_peak']:g} / $100"),
    ("summit", "Ladder", "Summit", "Finish a ladder at the goal", lambda s: s["lad_done"] >= 1, lambda s: f"${s['lad_peak']:g} / ${s['lad_goal']:g}"),
    ("sharp", "Sharp", "Sharp", "Beat the closing line on 10 picks", lambda s: s["clv_beat"] >= 10, lambda s: f"{min(s['clv_beat'], 10)}/10"),
]


def streaks_and_badges(potd_hist, ladder, prev_badges=None):
    """#13 Streaks and badges, rebuilt every run from the permanent records (nothing new to store).
    A badge keeps the date it was first earned."""
    graded = sorted([h for h in potd_hist if h.get("result") in ("W", "L")], key=lambda h: h["date"])
    cur_t, cur_n, best, run, best_end, run_start, best_start = None, 0, 0, 0, None, None, None
    for h in graded:
        if h["result"] == cur_t:
            cur_n += 1
        else:
            cur_t, cur_n = h["result"], 1
        if h["result"] == "W":
            run += 1
            run_start = run_start if run > 1 else h["date"]
            if run > best:
                best, best_end, best_start = run, h["date"], run_start
        else:
            run = 0
    months = {}
    for h in graded:
        m = months.setdefault(h["date"][:7], [0, 0]); m[0] += h["result"] == "W"; m[1] += 1
    good = {k: v for k, v in months.items() if v[1] >= 10 and v[0] / v[1] >= 0.55}
    bm = max(months.items(), key=lambda kv: (kv[1][0] / kv[1][1], kv[1][1])) if months else None
    bm = f"{bm[1][0]}-{bm[1][1] - bm[1][0]} ({bm[0]})" if bm else "-"
    steps = sorted(ladder.get("steps", []), key=lambda x: (x["attempt"], x["date"]))
    best_day, day_now, peak = 0, 0, ladder.get("start", 10)
    for st in steps:
        if st.get("result") == "W":
            peak = max(peak, st.get("bankroll_after") or 0)
    by_att = {}
    for st in steps:
        by_att.setdefault(st["attempt"], []).append(st)
    for att, rows in by_att.items():
        n = sum(1 for r in rows if r.get("result") == "W")
        best_day = max(best_day, n)
        if att == ladder.get("attempt"):
            day_now = n
    stats = {"potd_w": sum(1 for h in graded if h["result"] == "W"), "potd_best": best, "win_month": bool(good),
             "best_month": bm, "lad_best_day": best_day, "lad_peak": rnd(peak, 2), "lad_start": ladder.get("start", 10),
             "lad_goal": ladder.get("goal", 1000),
             "lad_done": sum(1 for a in ladder.get("attempts", []) if a.get("result") == "Completed"),
             "clv_beat": sum(1 for e in potd_hist + ladder.get("steps", []) if (e.get("clv") or {}).get("beat") and (e.get("clv") or {}).get("final"))}
    old = {b["id"]: b for b in (prev_badges or [])}
    today = datetime.now(ET).date().isoformat()
    badges = []
    for bid, group, name, desc, test, prog in BADGES:
        ok = bool(test(stats))
        earned = (old.get(bid) or {}).get("earned") or (today if ok else None)
        badges.append({"id": bid, "group": group, "name": name, "desc": desc, "earned": earned if (ok or earned) else None,
                       "progress": prog(stats)})
    return {"potd": {"current": {"type": cur_t, "n": cur_n} if cur_t else None,
                     "best_win": {"n": best, "start": best_start, "end": best_end},
                     "record": f"{stats['potd_w']}-{len(graded) - stats['potd_w']}"},
            "ladder": {"attempt": ladder.get("attempt"), "day": day_now, "bankroll": ladder.get("bankroll"),
                       "best_day": best_day, "peak": stats["lad_peak"], "completed": stats["lad_done"]},
            "badges": badges}


# ------------------------------------------------------------------ #9 Discord alerts
def discord_post(payload):
    if not DISCORD_WEBHOOK:
        return False
    try:
        r = requests.post(DISCORD_WEBHOOK, json=payload, timeout=20)
        if r.status_code in (200, 204):
            return True
        log(f"discord: HTTP {r.status_code}")
    except requests.RequestException as e:
        log(f"discord error: {e}")
    return False


def _pick_embed(title, p, color, extra=None):
    lines = [f"**{p['label']}** ({fmt_px(p['price'])})", f"{p['game'].replace('@', ' @ ')} - {p.get('time', '')}"]
    if p.get("score") is not None:
        lines.append(f"Confidence {round(p['score'])}/100" + (f" - win chance {p['p_win']:g}%" if p.get("p_win") is not None else ""))
    lines += ["", *[f"- {r}" for r in (p.get("reasons") or [])[:4]]]
    if extra:
        lines += ["", extra]
    return {"title": title, "url": SITE_URL, "description": "\n".join(lines)[:3900], "color": color}


def send_alerts(slot, potd_hist, ladder, streaks, slate, t_iso, now):
    """Morning run: post today's Pick of the Day and ladder step. 6 PM run: post again only if the pick changed.
    1 AM run: post last night's results. Each message goes out once (data/alerts.json). Needs the
    DISCORD_WEBHOOK_URL secret; without it this does nothing."""
    st = load(ALERTS_PATH, {})
    if not DISCORD_WEBHOOK:
        log("alerts: no DISCORD_WEBHOOK_URL secret set; skipping")
        return
    pick = next((h for h in potd_hist if h["date"] == t_iso), None)
    step = next((x for x in ladder.get("steps", []) if x["date"] == t_iso), None)
    cur = (streaks.get("potd") or {}).get("current")
    streak_txt = f"POTD record {streaks['potd']['record']}" + (f" - current streak {cur['type']}{cur['n']}" if cur else "")
    embeds = []
    if slot == "morning" and pick and st.get("pick_date") != t_iso:
        embeds.append(_pick_embed(f"Pick of the Day - {now.strftime('%a %b %-d')}", pick, 0xFF7A1A, streak_txt))
        if step:
            embeds.append(_pick_embed(f"Ladder Challenge - attempt #{step['attempt']}, day {step['day']}", step, 0x5CC8FF,
                                      f"Stake ${step['stake']:g} to win ${step['to_win']:g}"))
        if discord_post({"username": "PJ MODEL", "content": f"Today's picks are up: {SITE_URL}", "embeds": embeds}):
            st.update(pick_date=t_iso, pick_label=pick["label"], step_label=(step or {}).get("label"))
    elif slot in ("pretip", "tipcheck") and pick and st.get("pick_date") == t_iso and not pick.get("result"):
        live = next((g["status"] != "Upcoming" for g in slate if g["event_id"] == pick.get("eid")), False)
        if not live and pick["label"] != st.get("pick_label"):
            if discord_post({"username": "PJ MODEL", "content": "Pick of the Day updated with the latest lines and injuries:",
                             "embeds": [_pick_embed("Updated Pick of the Day", pick, 0xFF7A1A, f"Replaces: {st.get('pick_label')}")]}):
                st.update(pick_label=pick["label"])
    elif slot == "overnight":
        y = (now.date() - timedelta(days=1)).isoformat()
        if st.get("recap_date") != y:
            py = next((h for h in potd_hist if h["date"] == y and h.get("result")), None)
            ly = next((x for x in ladder.get("steps", []) if x["date"] == y and x.get("result")), None)
            word = {"W": "WON", "L": "LOST", "P": "PUSH"}
            out = []
            if py:
                out.append(f"Pick of the Day: **{py['label']}** - {word[py['result']]} ({py.get('actual', '')})")
            if ly:
                out.append(f"Ladder: **{ly['label']}** - {word[ly['result']]}" +
                           (f", bankroll now ${ladder.get('bankroll'):g}" if ly["result"] != "L" else f", attempt #{ly['attempt']} ends; a new ladder starts at ${ladder.get('start'):g}"))
            if out:
                new = [b["name"] for b in streaks.get("badges", []) if b.get("earned") == now.date().isoformat()]
                out += ["", streak_txt] + ([f"New badge: {', '.join(new)}"] if new else [])
                if discord_post({"username": "PJ MODEL", "embeds": [{"title": f"Results - {y}", "url": SITE_URL,
                                                                     "description": "\n".join(out), "color": 0x2ECC71 if (py or ly or {}).get("result") == "W" else 0xE74C3C}]}):
                    st.update(recap_date=y)
    save(ALERTS_PATH, st, indent=1)


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


# Alt-line value is easy to overstate: a projection far from DraftKings' main line, or a hot streak over 7-8
# games, makes long shots look like steals. So the projection is pulled halfway to the main line, cards where
# they disagree by more than BET_EDGE_PCT_MAX are skipped (the model is probably missing news), and the recent
# hit rate is smoothed with ALT_HIT_PRIOR games' worth of the model's own number.
ALT_ANCHOR = 0.5
ALT_HIT_PRIOR = 10


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
            raw_mean = pr["model"] if pr else sum(vals[:10]) / len(vals[:10])
            main = pr["dk"] if pr else None
            if main and abs(raw_mean - main) / main > BET_EDGE_PCT_MAX:
                continue
            mean = main + (raw_mean - main) * ALT_ANCHOR if main else raw_mean
            ladder = []
            for line, price in rungs:
                if price < -1000 or price > 3000:
                    continue
                pm = prob_over(stat, mean, line, vals)
                hit = sum(1 for v in vals if v > line)
                p = 0.6 * pm + 0.4 * (hit + ALT_HIT_PRIOR * pm) / (len(vals) + ALT_HIT_PRIOR)
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
            out.append({"player": player, "team": hist[-1]["t"], "game": g["id"] if g else "", "eid": eid, "time": g["time"] if g else "",
                        "stat": stat, "model": rnd(mean, 1), "model_raw": rnd(raw_mean, 1), "main_line": pr["dk"] if pr else None,
                        "avg5": rnd(sum(vals[:5]) / len(vals[:5]), 1), "avg15": rnd(sum(vals) / len(vals), 1),
                        "last10": vals[:10][::-1], "rungs": ladder, "suggested": [r["line"] for r in pick],
                        "best_ev": best["ev"], "value_rungs": len(value)})
    out.sort(key=lambda x: (-x["value_rungs"], -x["best_ev"]))
    return out[:24]


# ------------------------------------------------------------------ permanent bet ledger
def _status(r):
    return "Pending" if not r else {"W": "Won", "L": "Lost", "P": "Push"}.get(r, r)


LEDGER_FIELDS = ("player", "team", "opp", "stat", "dir", "line", "price", "model", "edge")


def _pick_fields(h):
    """Structured fields for a Pick of the Day / ladder entry, so every record sorts and filters the same way."""
    t = h.get("type", "Prop")
    team = h.get("team")
    opp = (h.get("away") if team == h.get("home") else h.get("home")) if team and t != "Total" else None
    return {"type": t, "player": h.get("player"), "team": team if t != "Total" else None, "opp": opp,
            "stat": h.get("stat") if t == "Prop" else t, "dir": h.get("side"), "line": h.get("line"),
            "price": h.get("price"), "model": h.get("model"), "edge": h.get("edge"), "edge_pct": h.get("edge_pct"),
            "best": (h.get("best_over") if h.get("side") == "Over" else h.get("best_under")) if t == "Prop" else None}


def model_row(b):
    r = b.get("result")
    return {"id": f"model|{b['date']}|{b['game']}|{b.get('key') or b['side']}", "source": "Model bet",
            "date": b["date"], "game": b["game"], "market": b.get("market"), "type": b.get("type") or ("Prop" if b.get("market") == "Player Prop" else b.get("market")),
            "pick": b["side"], **{k: b.get(k) for k in LEDGER_FIELDS},
            "confidence": b.get("confidence"), "win_chance": b.get("win_chance"), "price_book": b.get("price_book"),
            "result": r, "status": _status(r), "actual": b.get("actual"), "units": b.get("units"),
            "why": f"Model {b.get('model')} vs DraftKings {b.get('line')} (edge {b.get('edge')})" if b.get("line") is not None else f"Model edge of {b.get('edge')} vs the DraftKings line"}


ALT_SPLIT = {1: [1.0], 2: [0.6, 0.4], 3: [0.5, 0.3, 0.2]}   # same split the Alt-Line page shows


def track_alt_ladders(alt_ladders, alt_date, slate, plog, now):
    """Save each day's suggested alt-line ladders (1 unit split across the rungs, as the page shows) and grade
    them after the game. A ladder can change until its game tips off; then it is locked. Returns the log."""
    store = load(ALT_LOG, {"note": "Every suggested alt-line ladder: 1 unit split across its rungs (50/30/20, or 60/40), graded after the game. Never trimmed.",
                           "ladders": []})
    L = store["ladders"]
    status = {g["event_id"]: g["status"] for g in slate}
    stamp = now.strftime("%Y-%m-%d %I:%M %p ET")
    upcoming = lambda x: status.get(x.get("eid"), "Final") == "Upcoming"
    if alt_ladders and alt_date:
        live = set()
        by_game = {g["id"]: g["event_id"] for g in slate}
        for a in alt_ladders:
            a.setdefault("eid", by_game.get(a.get("game")))         # ladders saved before eid was stored
            if not a.get("suggested") or not a.get("eid"):
                continue
            key = f"{alt_date}|{a['player']}|{a['stat']}"
            live.add(key)
            cur = next((x for x in L if x["id"] == key), None)
            if cur and (cur.get("result") or not upcoming(cur)):
                continue                                           # locked at tip-off
            rungs = sorted([r for r in a["rungs"] if r["line"] in a["suggested"]], key=lambda r: r["line"])[:3]
            split = ALT_SPLIT[len(rungs)]
            entry = {"id": key, "date": alt_date, "player": a["player"], "team": a.get("team"), "stat": a["stat"],
                     "game": a["game"], "eid": a["eid"], "time": a.get("time"), "main_line": a.get("main_line"), "model": a.get("model"),
                     "rungs": [{"line": r["line"], "price": r["price"], "p": r["p"], "stake": split[i]} for i, r in enumerate(rungs)],
                     "saved_at": (cur or {}).get("saved_at", stamp), "updated_at": stamp}
            if cur:
                L[L.index(cur)] = entry
            else:
                L.append(entry)
        # a ladder that was dropped by a later rebuild, before its game tipped, was never a bet
        L[:] = [x for x in L if x.get("result") or x["date"] != alt_date or x["id"] in live or not upcoming(x)]
    stale = (now.date() - timedelta(days=3)).isoformat()
    for x in L:
        if x.get("result"):
            continue
        g = plog["games"].get(x["eid"])
        if not g:
            if x["date"] < stale:
                x.update(result="P", actual="No result (void)", units=0.0, graded_at=stamp)
            continue
        row = next((r for r in g["players"] if norm(r["n"]) == norm(x["player"])), None)
        if row is None:
            x.update(result="P", actual="Did not play (void)", units=0.0, hit=0, graded_at=stamp)
            continue
        v = STAT_VAL[x["stat"]](row)
        hits = [r for r in x["rungs"] if v > r["line"]]
        units = rnd(sum(r["stake"] * decimal(r["price"]) for r in hits) - 1, 2)
        x.update(value=v, hit=len(hits), units=units, result="W" if units > 0 else "L" if units < 0 else "P",
                 actual=f"{v:g} {x['stat']} · {len(hits)} of {len(x['rungs'])} rungs hit", graded_at=stamp)
    L.sort(key=lambda x: (x["date"], x["player"]), reverse=True)
    save(ALT_LOG, store, indent=1)
    return L


def alt_row(x):
    names = ["Safe", "Middle", "Long shot"] if len(x["rungs"]) == 3 else ["Safe", "Long shot"] if len(x["rungs"]) == 2 else ["Single"]
    r = x.get("result")
    return {"id": f"alt|{x['id']}", "source": "Alt Ladder", "date": x["date"], "game": x["game"], "market": "Alt Ladder",
            "type": "Alt Ladder", "player": x["player"], "team": x.get("team"), "stat": x["stat"], "dir": "Over",
            "line": x["rungs"][0]["line"] if x["rungs"] else None, "price": None, "model": x.get("model"),
            "pick": f"{x['player']} {x['stat']} ladder: " + " / ".join(f"{math.ceil(r['line'])}+ ({fmt_px(r['price'])})" for r in x["rungs"]),
            "win_chance": x["rungs"][0]["p"] if x["rungs"] else None, "result": r, "status": _status(r),
            "actual": x.get("actual"), "units": x.get("units"),
            "why": " | ".join(f"{n}: {math.ceil(r['line'])}+ {x['stat']} at {fmt_px(r['price'])}, {round(r['stake'] * 100)}% of the stake, our chance {r['p']:g}%"
                              for n, r in zip(names, x["rungs"])),
            "picked_at": x.get("saved_at"), "graded_at": x.get("graded_at")}


def ledger_rows(settled_bets, potd_hist, ladder, pending_bets=(), alt_log=()):
    rows = [model_row(b) for b in pending_bets] + [model_row(b) for b in settled_bets] + [alt_row(x) for x in alt_log if x.get("rungs")]
    for h in potd_hist:
        rows.append({"id": f"potd|{h['date']}", "source": "Pick of the Day", "date": h["date"], "game": h["game"],
                     "market": h.get("type", "Prop"), "pick": h.get("label") or f"{h.get('player')} {h.get('side')} {h.get('line')} {h.get('stat')}",
                     **_pick_fields(h), "confidence": h.get("score"), "win_chance": h.get("p_win"),
                     "result": h.get("result"), "status": _status(h.get("result")), "actual": h.get("actual"),
                     "units": h.get("units"), "why": " | ".join(x for x in (h.get("reasons") or []) if x),
                     "picked_at": h.get("picked_at"), "graded_at": h.get("graded_at"), "clv": h.get("clv"),
                     "open": [h.get("open_line"), h.get("open_price")], "close": [h.get("close_line"), h.get("close_price")]})
    for st in ladder.get("steps", []):
        r = st.get("result")
        rows.append({"id": f"ladder|{st['attempt']}|{st['day']}|{st['date']}", "source": "Ladder Challenge",
                     "date": st["date"], "game": st["game"], "market": st.get("type", "Prop"), "pick": st.get("label"),
                     **_pick_fields(st), "confidence": st.get("score"),
                     "attempt": st["attempt"], "day": st["day"], "stake": st.get("stake"), "to_win": st.get("to_win"),
                     "bankroll_after": st.get("bankroll_after"), "win_chance": st.get("p_win"),
                     "result": r, "status": _status(r), "actual": st.get("actual"),
                     "units": (rnd(st["stake"] * (decimal(st["price"]) - 1), 2) if r == "W" else -st["stake"] if r == "L" else 0.0) if r else None,
                     "why": " | ".join([st.get("why_ladder", "")] + [x for x in (st.get("reasons") or []) if x]).strip(" |"),
                     "picked_at": st.get("picked_at"), "graded_at": st.get("graded_at"), "clv": st.get("clv"),
                     "open": [st.get("open_line"), st.get("open_price")], "close": [st.get("close_line"), st.get("close_price")]})
    return rows


def update_ledger(settled_bets, potd_hist, ladder, now, pending_bets=(), open_games=(), alt_log=()):
    """Upsert every bet into data/records/bets_YYYY-MM.json and rebuild summary.json. Graded bets are never
    deleted. The one exception: a Pending model pick on a game that has not tipped, which the latest lines
    no longer support, is taken back off (it was never a bet)."""
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    by_month = {}
    live_ids = {model_row(b)["id"] for b in pending_bets}
    open_games = set(open_games)
    live_alt = {f"alt|{x['id']}" for x in alt_log}
    for r in ledger_rows(settled_bets, potd_hist, ladder, pending_bets, alt_log):
        by_month.setdefault(r["date"][:7], []).append(r)
    for (d, gl) in open_games:
        by_month.setdefault(d[:7], [])
    for month, rows in by_month.items():
        path = RECORDS_DIR / f"bets_{month}.json"
        cur = {x["id"]: x for x in load(path, {"bets": []}).get("bets", [])}
        for k in [k for k, x in cur.items() if x.get("source") == "Model bet" and not x.get("result")
                  and (x["date"], x["game"]) in open_games and k not in live_ids]:
            del cur[k]
        for k in [k for k, x in cur.items() if x.get("source") == "Alt Ladder" and not x.get("result") and k not in live_alt]:
            del cur[k]                                    # dropped before tip-off: never a bet
        for r in rows:
            cur[r["id"]] = {**cur.get(r["id"], {}), **{k: v for k, v in r.items() if v is not None or k not in cur.get(r["id"], {})}}
        order = {"Pick of the Day": 0, "Ladder Challenge": 1, "Model bet": 2}
        bets = sorted(cur.values(), key=lambda x: (x["date"], -order.get(x["source"], 9)), reverse=True)
        save(path, {"month": month, "updated": now.strftime("%Y-%m-%d %I:%M %p ET"), "count": len(bets), "bets": bets}, indent=1)
    # summary across every month on disk
    summary, recent, model_n = {}, [], [0]
    for f in sorted(RECORDS_DIR.glob("bets_*.json"), reverse=True):
        data = load(f, {"bets": []})
        for b in data["bets"]:
            for key in (("All", b["source"]), (data.get("month", f.stem[5:]), b["source"])):
                t = summary.setdefault(key[0], {}).setdefault(key[1], {"bets": 0, "won": 0, "lost": 0, "push": 0, "pending": 0, "units": 0.0})
                t["bets"] += 1
                t[{"W": "won", "L": "lost", "P": "push"}.get(b.get("result"), "pending")] += 1
                if b["source"] != "Ladder Challenge":
                    t["units"] = rnd(t["units"] + (b.get("units") or 0), 2)
        # the page gets every Pick of the Day and ladder step plus the newest 300 model picks;
        # the Pick Log tab loads whole months from data/records/ when it needs more
        recent += [b for b in data["bets"] if b["source"] != "Model bet"]
        if model_n[0] < 300:
            mb = [b for b in data["bets"] if b["source"] == "Model bet"][:300 - model_n[0]]
            recent += mb; model_n[0] += len(mb)
    save(RECORDS_DIR / "summary.json", {"updated": now.strftime("%Y-%m-%d %I:%M %p ET"),
                                        "note": "Units are 1 unit per bet for Model bets and Pick of the Day. Ladder steps are tracked in dollars on the ladder itself.",
                                        "by_month": summary}, indent=1)
    recent.sort(key=lambda b: (b["date"], {"Pick of the Day": 2, "Ladder Challenge": 1}.get(b["source"], 0)), reverse=True)
    months = sorted({f.stem[5:] for f in RECORDS_DIR.glob("bets_*.json")}, reverse=True)
    return {"summary": summary, "bets": recent, "months": months}


CALIB_PATH = ROOT / "data" / "calibration.json"
PAUSE_MIN_BETS = 150     # a market needs this many graded model leans before it can be paused
PAUSE_ROI = -8.0         # ... and an ROI this bad (%, 1 unit per lean at the posted price)


def calib_record(cal, market, p, res, price):
    """Add one graded model lean. p = model's chance its side wins; res W/L/P; price = posted DK price."""
    if res == "P" or p is None:
        return
    p = max(0.5, min(0.999, p))
    b = str(min(90, int(p * 100) // 5 * 5))                # 50, 55, ... 90 (90 = 90%+)
    win = res == "W"
    units = ((price / 100 if price > 0 else 100 / -price) if price else 100 / 110) if win else -1.0
    for key in ("ALL", market):
        m = cal["buckets"].setdefault(key, {})
        r = m.setdefault(b, {"n": 0, "w": 0, "p": 0.0})
        r["n"] += 1; r["w"] += win; r["p"] = rnd(r["p"] + p, 3)
    r = cal["markets"].setdefault(market, {"n": 0, "w": 0, "p": 0.0, "units": 0.0})
    r["n"] += 1; r["w"] += win; r["p"] = rnd(r["p"] + p, 3); r["units"] = rnd(r["units"] + units, 2)


ACC_MIN_GAMES = 10   # game lines need this many compared games before the accuracy pause can kick in
ACC_WINDOW = 40      # ... measured over the most recent games


def paused_markets(cal, accuracy=None):
    """#8 auto-pause: markets with a big enough sample that keep losing are left out of Pick of the Day and
    the ladder until they recover. Game lines are also paused while the model misses actual results by more
    than DraftKings does (a spread or total pick only makes sense if the model beats the market's number).
    Moneylines use the same margin model as spreads, so they pause with them. Shown on the Model tab."""
    out = {}
    acc = (accuracy or [])[:ACC_WINDOW]
    if len(acc) >= ACC_MIN_GAMES:
        miss = lambda f: sum(abs(f(x)) for x in acc) / len(acc)
        ms, ds = miss(lambda x: -x["model_spread"] - x["actual_margin"]), miss(lambda x: -x["dk_spread"] - x["actual_margin"])
        mt, dt = miss(lambda x: x["model_total"] - x["actual_total"]), miss(lambda x: x["dk_total"] - x["actual_total"])
        if ms > ds:
            out["Spread"] = out["Moneyline"] = f"model misses the margin by {ms:.1f} pts vs DraftKings {ds:.1f} (last {len(acc)} games)"
        if mt > dt:
            out["Total"] = f"model misses the total by {mt:.1f} pts vs DraftKings {dt:.1f} (last {len(acc)} games)"
    for mk, r in (cal or {}).get("markets", {}).items():
        if r["n"] >= PAUSE_MIN_BETS:
            roi = 100 * r["units"] / r["n"]
            if roi <= PAUSE_ROI:
                out[mk] = rnd(roi, 1)
    return out


def _px(v):
    try:
        return int(num(v)) if v not in (None, "N/A") else None
    except (TypeError, ValueError):
        return None


def prop_bet(p, label, date):
    """One model prop pick in the shape the records store (same shape before and after grading)."""
    over = p["model"] > p["dk"]
    px, book = bet_price(p, over)
    return {"key": f'{p["player"]}|{p["stat"]}', "date": date, "game": label, "type": "Prop", "market": "Player Prop",
            "player": p["player"], "team": p.get("team"), "opp": p.get("opp"), "stat": p["stat"],
            "dir": "Over" if over else "Under", "line": p["dk"], "price": px, "price_book": book,
            "model": p["model"], "edge": rnd(p["model"] - p["dk"]),
            "confidence": p.get("conf"), "win_chance": p.get("win_chance"), "win_raw": p.get("win_raw"), "hits": p.get("hits"),
            "side": f'{p["player"]} {"Over" if over else "Under"} {p["dk"]:g} {p["stat"]}'}


def game_bets(g, label, date):
    """Model spread / total picks for one game (g has home, away, model, dk)."""
    dk, out = g.get("dk") or {}, []
    try:
        hs = float(dk["spread"])
    except (KeyError, TypeError, ValueError):
        return out
    sedge = hs - g["model"]["spread"]
    if abs(sedge) >= BET_EDGE_GAME and _px(dk.get("spread_px_home" if sedge > 0 else "spread_px_away")) is not None:
        hp = sedge > 0
        team, opp = (g["home"], g["away"]) if hp else (g["away"], g["home"])
        line = hs if hp else -hs
        out.append({"key": "Spread", "date": date, "game": label, "type": "Spread", "market": "Spread", "team": team, "opp": opp,
                    "stat": "Spread", "dir": "Cover", "line": line, "price": _px(dk.get("spread_px_home" if hp else "spread_px_away")),
                    "model": rnd(-g["model"]["spread"] if hp else g["model"]["spread"]), "edge": rnd(sedge),
                    "side": f"{team} {sgn(line)}", "_hp": hp, "_hs": hs})
    try:
        tl = float(dk["total"])
    except (KeyError, TypeError, ValueError):
        return out
    if tl and abs(g["model"]["total"] - tl) >= BET_EDGE_GAME and _px(dk.get("over_px" if g["model"]["total"] > tl else "under_px")) is not None:
        ov = g["model"]["total"] > tl
        out.append({"key": "Total", "date": date, "game": label, "type": "Total", "market": "Total", "team": None, "opp": None,
                    "stat": "Total", "dir": "Over" if ov else "Under", "line": tl, "price": _px(dk.get("over_px" if ov else "under_px")),
                    "model": g["model"]["total"], "edge": rnd(g["model"]["total"] - tl),
                    "side": f'{"Over" if ov else "Under"} {tl:g}'})
    return out


def prop_strength(p, hist):
    """Confidence (0-100, same formula as Pick of the Day: recent hit rates + edge size) and calibrated win
    chance for a prop's lean side, from games before tonight only. Saved on the prop when it is picked."""
    if not p.get("dk") or p.get("p_over") is None:
        return
    over = p["model"] > p["dk"]
    vals = [STAT_VAL[p["stat"]](r) for r in (hist or [])[::-1] if r["min"] >= 5] if p["stat"] in STAT_VAL else []
    ok = (lambda v: v > p["dk"]) if over else (lambda v: v < p["dk"])
    (a5, n5), (a10, n10), (a15, n15) = _rate(vals, ok, 5), _rate(vals, ok, 10), _rate(vals, ok, 15)
    if not n15:
        return
    rel = abs(p["model"] - p["dk"]) / p["dk"]
    p["hits"] = {"l5": [a5, n5], "l10": [a10, n10], "l15": [a15, n15]}
    p["conf"] = rnd((0.55 * (0.5 * a10 / n10 + 0.25 * a5 / n5 + 0.25 * a15 / n15) + 0.45 * min(rel / 0.25, 1)) * 100, 1)
    pm = p["p_over"] if over else 1 - p["p_over"]
    raw = 0.6 * pm + 0.4 * a15 / n15
    p["win_raw"] = rnd(raw * 100, 1)
    p["win_chance"] = rnd(calibrate(raw) * 100, 1)


def bet_price(p, over):
    """DraftKings price for the side, else the best price at another book: (price, book)."""
    px = p.get("over") if over else p.get("under")
    if px is not None:
        return px, "draftkings"
    best = p.get("best_over" if over else "best_under")
    return (best[0], best[1]) if best and best[0] is not None else (None, None)


def prop_qualifies(p):
    """Does this prop's edge clear the bar to count as a model bet? Big enough in points AND as a share of
    the line (stricter for Unders), but not so big that the model is probably missing news."""
    if not p.get("dk") or bet_price(p, p["model"] > p["dk"])[0] is None:
        return False                                   # no price anywhere: it can't be bet, so it isn't one
    e = p["model"] - p["dk"]
    rel = abs(e) / p["dk"]
    return abs(e) >= BET_EDGE_PROP and (BET_EDGE_PCT_OVER if e > 0 else BET_EDGE_PCT_UNDER) <= rel <= BET_EDGE_PCT_MAX


def select_prop_bets(props):
    """One direction per player per game. Betting Over on one of his stats and Under on another (say PTS Under
    but PRA Over) mostly cancels out, so each player keeps only the side his edges point to overall
    (summed as a share of each line). Returns the set of (player, stat) that count as bets."""
    by_player = {}
    for p in props:
        if prop_qualifies(p):
            by_player.setdefault(norm(p["player"]), []).append(p)
    keep = set()
    for pl, ps in by_player.items():
        lean = sum((p["model"] - p["dk"]) / p["dk"] for p in ps if p["dk"])
        over = lean >= 0
        for p in ps:
            if (p["model"] > p["dk"]) == over:
                keep.add((pl, p["stat"]))
    return keep


def pending_model_bets(props, games):
    """Every model pick on games that have not tipped yet, saved to the records as Pending right away.
    Same rule as grading: props that pass prop_qualifies (one side per player), spreads/totals with BET_EDGE_GAME+."""
    rows = []
    up = {r["id"]: r for r in games if r.get("status") == "Upcoming"}
    for r in up.values():
        rows += game_bets(r, f'{r["away"]} @ {r["home"]}', r["date"])
    by_game = {}
    for p in props:
        if up.get(p["game"]):
            by_game.setdefault(p["game"], []).append(p)
    for gid, ps in by_game.items():
        r, keep = up[gid], select_prop_bets(ps)
        for p in ps:
            if (norm(p["player"]), p["stat"]) in keep:
                rows.append(prop_bet(p, f'{r["away"]} @ {r["home"]}', r["date"]))
    for b in rows:
        b.pop("_hp", None); b.pop("_hs", None)
    return rows


def settle(pending, plog, state, cal=None):
    """Score frozen predictions for games that have finished. Updates state in place."""
    names = {}
    for eid, g in plog["games"].items():
        names[eid] = {norm(p["n"]): p for p in g["players"]}
    done = []
    state["_n_before"] = len(state["settled_bets"])
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
            for b in game_bets(pg, label, lg["date"]):
                if b["type"] == "Spread":
                    cover = margin + hs
                    res = "P" if cover == 0 else ("W" if (cover > 0) == b["_hp"] else "L")
                    act = f"{lg['away_pts']}-{lg['home_pts']}"
                else:
                    res = "P" if total == b["line"] else ("W" if (total > b["line"]) == (b["dir"] == "Over") else "L")
                    act = f"Total {total}"
                b.pop("_hp", None); b.pop("_hs", None)
                bets.append({**b, "result": res, "actual": act})
            if cal is not None and sedge:
                hp = sedge > 0
                mm = -pg["model"]["spread"] if hp else pg["model"]["spread"]          # model margin for our side
                ln = hs if hp else -hs
                cov = margin + hs
                if cov:
                    calib_record(cal, "Spread", phi((mm + ln) / GAME_SD), "W" if (cov > 0) == hp else "L",
                                 dk.get("spread_px_home" if hp else "spread_px_away"))
            tedge = pg["model"]["total"] - dk["total"]
            if cal is not None and tedge and total != dk["total"]:
                ov = tedge > 0
                po = 1 - phi((dk["total"] - pg["model"]["total"]) / 17.0)
                calib_record(cal, "Total", po if ov else 1 - po, "W" if (total > dk["total"]) == ov else "L",
                             dk.get("over_px" if ov else "under_px"))
        keep = select_prop_bets(pending["props"].get(eid, []))
        for p in pending["props"].get(eid, []):
            row = names[eid].get(norm(p["player"]))
            is_bet = (norm(p["player"]), p["stat"]) in keep
            if not row:
                # did not play: the pick is void (refunded at the book), so it is closed out as a push
                if is_bet:
                    bets.append({**prop_bet(p, label, lg["date"]), "result": "P", "actual": "Did not play (void)"})
                continue
            actual = {"PTS": row["pts"], "REB": row["reb"], "AST": row["ast"], "3PM": row["tpm"],
                      "BLK": row["blk"], "STL": row["stl"], "PRA": row["pts"] + row["reb"] + row["ast"],
                      "PR": row["pts"] + row["reb"], "PA": row["pts"] + row["ast"],
                      "RA": row["reb"] + row["ast"]}.get(p["stat"])
            if actual is None:
                continue
            key = f'{p["player"]}|{p["stat"]}'
            old = state["residuals"]["player"].get(key, 0.0)
            state["residuals"]["player"][key] = rnd((1 - RES_BETA) * old + RES_BETA * (actual - p.get("base", p["model"])), 2)
            edge = p["model"] - p["dk"]
            if cal is not None and edge and p.get("p_over") is not None and actual != p["dk"]:
                lean_over = edge > 0
                calib_record(cal, p["stat"], p["p_over"] if lean_over else 1 - p["p_over"],
                             "W" if (actual > p["dk"]) == lean_over else "L", p.get("over") if lean_over else p.get("under"))
                if p.get("win_raw") is not None and (p["model"] > p["dk"]) == lean_over:
                    blend_record(cal, p["win_raw"] / 100, "W" if (actual > p["dk"]) == lean_over else "L")
            if is_bet:
                b = prop_bet(p, label, lg["date"])
                res = "P" if actual == p["dk"] else ("W" if (actual > p["dk"]) == (b["dir"] == "Over") else "L")
                bets.append({**b, "result": res, "actual": f'{actual:g} {p["stat"]}'})
        if dk:
            state["accuracy"].insert(0, {"date": lg["date"], "game": label, "model_spread": pg["model"]["spread"],
                                         "dk_spread": float(dk["spread"]), "actual_margin": margin,
                                         "model_total": pg["model"]["total"], "dk_total": dk["total"], "actual_total": total})
        for bet in bets:
            res, price = bet["result"], bet.get("price")
            win = (price / 100 if price and price > 0 else 100 / -price if price else WIN_UNITS)
            units = rnd(win, 2) if res == "W" else (-1.0 if res == "L" else 0.0)
            state["settled_bets"].insert(0, {**bet, "units": rnd(units, 2)})
            rec = state["record"]
            rec["wins"] += res == "W"
            rec["losses"] += res == "L"
            rec["units"] = rnd(rec["units"] + units, 2)
        state["residuals"]["games_learned"] += 1
        pg["settled"] = True
        done.append(eid)
    state["new_bets"] = state["settled_bets"][:len(state["settled_bets"]) - state.get("_n_before", 0)]
    state["settled_bets"] = state["settled_bets"][:500]
    state["accuracy"] = state["accuracy"][:1000]
    log(f"settled {len(done)} games")


# -------------------------------------------------------------------- main
def main():
    now = datetime.now(ET)
    global CREDITS
    CREDITS = Credits(now)
    slot = "tipcheck" if TIP_CHECK else "overnight" if now.hour < 6 else ("pretip" if now.hour >= 17 else "morning")
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
    cal = load(CALIB_PATH, {"note": "Every graded model lean (all posted props, spreads and totals), bucketed by the model's win chance.",
                            "buckets": {}, "markets": {}})
    settle(pending, plog, state, cal)
    global CAL_MAP
    blend_n = sum(r["n"] for r in (cal.get("blend") or {}).values())
    if blend_n >= CAL_BLEND_MIN:
        CAL_MAP = build_cal_map({"buckets": {"ALL": cal["blend"]}})
        log(f"calibration: built from {blend_n} graded blended win chances")
    else:
        CAL_MAP = build_cal_map(cal)
        log(f"calibration: built from raw model leans ({blend_n}/{CAL_BLEND_MIN} blended win chances graded so far)")
    log("calibration map: " + ", ".join(f"{x * 100:.0f}%->{y * 100:.0f}%" for x, y in CAL_MAP))
    state["residuals"]["alpha"] = rnd(0.12 + 0.33 * math.exp(-state["residuals"]["games_learned"] / 60), 3)

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
    if upcoming and slot not in ("overnight", "tipcheck") and ODDS_KEY:
        CREDITS.spend(f"{ODDS}/events", 0, "balance check")   # free call: reads the key's real balance before any paid call
    gml = odds_game_lines() if (upcoming and slot not in ("overnight", "tipcheck")) else {}   # 1 AM lines move before anyone bets them
    def tip(g):
        m = re.match(r"(\d+):(\d+) (AM|PM)", g["time"])
        return (int(m.group(1)) % 12 + (12 if m.group(3) == "PM" else 0)) * 60 + int(m.group(2)) if m else 9999
    todays_up = sorted([g for g in upcoming if g["date"] == t_iso], key=tip)
    if gml:   # props only for games whose game lines are posted (props come out after them)
        todays_up = [g for g in todays_up if (NAME_FIX.get(g["names"][g["away"]], g["names"][g["away"]]), NAME_FIX.get(g["names"][g["home"]], g["names"][g["home"]])) in gml] or todays_up
    prop_lines = odds_props(todays_up, idx) if (todays_up and slot not in ("overnight", "tipcheck")) else {}
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
            if slot == "tipcheck" and frozen and frozen.get("dk"):
                dk = frozen["dk"]                                # free check: keep the lines from the last paid run
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
            for player, stat, line, over_px, under_px, *extra in prop_lines[eid]:
                extra = extra[0] if extra else {}
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
                mdl = with_residual(state, hist[-1]["n"], stat, proj[stat])
                fresh.append({"player": hist[-1]["n"], "team": team, "opp": opp, "game": g["id"], "stat": stat,
                              "model": mdl, "base": proj[stat], "matchup": proj["_matchup"][stat],
                              "injury": proj["_injury"][stat], "inj_out": sorted(set(bst["out"])) if bst else [],
                              "sched": proj["_sched"], "sched_notes": proj["_sched_notes"],
                              "p_over": p_over_for(stat, mdl, line, hist),
                              "dk": line, "over": over_px, "under": under_px, **extra})
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
                    p.update(opp=opp, model=with_residual(state, p["player"], p["stat"], proj[p["stat"]]), base=proj[p["stat"]],
                             matchup=proj["_matchup"][p["stat"]],
                             injury=proj["_injury"][p["stat"]], inj_out=sorted(set(bst["out"])) if bst else [],
                             sched=proj["_sched"], sched_notes=proj["_sched_notes"],
                             p_over=p_over_for(p["stat"], p["model"], p["dk"], hist))
        for p in pending["props"].get(eid, []):
            prop_strength(p, idx.get(norm(p["player"])))
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
    paused = paused_markets(cal, state["accuracy"])
    pick, new_alts, all_cands = pick_of_the_day(props, games, slate, idx, plog, injuries, t_iso, paused)
    add_win_probs(all_cands)
    # Pick of the Day and the ladder are always two different plays. The Pick of the Day gets first choice,
    # unless today's ladder step already locked (its game tipped first): then the Pick of the Day steps around it.
    lad_state = load(LADDER_PATH, {})
    lad_today = next((x for x in lad_state.get("steps", []) if x["date"] == t_iso), None)
    st_now = {g["event_id"]: g["status"] for g in slate}
    lad_locked = bool(lad_today) and bool(lad_today.get("result") or any(st_now.get(e, "Final") != "Upcoming" for e in step_eids(lad_today)))
    lad_keys = step_keys(lad_today) if lad_locked else set()
    # The Pick of the Day must be a winning bet on average at honest (calibrated) win chances: the highest-scoring
    # bet with positive expected value. If nothing on the board has value, there is no pick today, on purpose.
    ev_ok = lambda c: c.get("ev", -1) > 0
    if pick and (not ev_ok(pick) or bet_key(pick) in lad_keys):
        why = "locked ladder bet" if bet_key(pick) in lad_keys else f"{pick['ev']:+g}% expected value at honest odds"
        log(f"potd: passed on {pick['label']} ({why})")
        pick = next((c for c in all_cands if ev_ok(c) and bet_key(c) not in lad_keys), None)
    if pick:
        new_alts = [c for c in new_alts if bet_key(c) != bet_key(pick)]
        imp = implied(pick["price"]) * 100
        pick["reasons"] = pick["reasons"][:1] + [f"Honest win chance {pick['p_win']:g}% vs {imp:.0f}% implied by {fmt_px(pick['price'])}: "
                                                 f"{pick['ev']:+g}% expected value (raw model said {pick.get('p_win_raw', pick['p_win']):g}%)"] + pick["reasons"][1:]
    top = CAL_MAP[-1][1] if CAL_MAP else None
    board = {"date": t_iso, "qualified": len(all_cands), "positive": sum(1 for c in all_cands if ev_ok(c)),
             "top_win": rnd(top * 100, 1) if top else None,
             "note": ("The honest numbers currently say no bet beats the book's cut at normal prices; the model needs to "
                      "prove itself on more graded games before it can pick again." if top and top * decimal(-110) <= 1 else None)}
    log(f"board: {board['qualified']} qualifying bets, {board['positive']} with positive expected value")
    if not locked:
        potd_alts = new_alts
        potd_hist = [h for h in potd_hist if h["date"] != t_iso]
        if pick:
            pick["picked_at"] = now.strftime("%Y-%m-%d %I:%M %p ET")
            potd_hist.insert(0, carry_open(pick, todays))
    # 3c. Ladder challenge (uses the same qualifying bets, but picks the safest one with a real payout)
    potd_today = next((h for h in potd_hist if h["date"] == t_iso), None)
    ladder = run_ladder(lad_state, plog, all_cands, slate, t_iso, now, avoid={bet_key(potd_today)} - {None},
                        avoid_games={potd_today["eid"]} if potd_today and potd_today.get("eid") else set())
    track_clv(potd_hist[:5] + [x for s in ladder["steps"][:5] for x in (s.get("legs") or [s])], games, props, slate, now)
    streaks = streaks_and_badges(potd_hist, ladder, (prev.get("streaks") or {}).get("badges"))

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

    alt_log = track_alt_ladders(alt_ladders, alt_date, slate, plog, now)
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
                            "Bets are scored at 1 unit risked at the posted DraftKings price (another book's price when DraftKings has none; a bet with no price anywhere isn't counted. Bets before Nov 5 used -110 when no price was posted). Calibration grades every model lean; a market with 150+ graded leans and an ROI of -8% or worse is paused from Pick of the Day and the ladder until it recovers. Spreads, totals and moneylines are also paused while the model misses actual results by more than DraftKings does (last 40 games, 10+ needed)",
                            "Honest win chances: every win chance shown for a Pick of the Day or ladder pick is calibrated against how the model's past leans actually did, so a pick the raw model rates 80% shows what picks like it have really hit. The map updates after every run"],
            "prediction_history": state["prediction_history"], "notes": note,
            "settled_bets": state["settled_bets"], "record": state["record"],
            "accuracy": state["accuracy"], "standings": standings,
            "potd_history": potd_hist[:400], "potd_alts": potd_alts, "board": board,
            "ladder": {**{k: v for k, v in ladder.items() if k != "steps"}, "steps": ladder["steps"][:120]},
            "alt_ladders": alt_ladders, "alt_date": alt_date,
            "streaks": streaks,
            "calibration": {**cal, "paused": paused, "pause_rule": {"min_bets": PAUSE_MIN_BETS, "roi": PAUSE_ROI}},
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
    save(CALIB_PATH, cal, indent=1)
    CREDITS.save()
    ladder["steps"] = ladder["steps"][:1000]
    save(LADDER_PATH, ladder, indent=1)
    open_bets = pending_model_bets(props, games)
    open_games = {(r["date"], f'{r["away"]} @ {r["home"]}') for r in games if r.get("status") == "Upcoming"}
    records = update_ledger(state.get("new_bets", []) + state["settled_bets"], potd_hist, ladder, now, open_bets, open_games, alt_log)
    data["records"] = records
    save(POTD_LOG, {"note": "Every Pick of the Day with its reasons and result. Never trimmed.", "picks": potd_hist}, indent=1)
    save(OUT, data, indent=1)
    try:
        send_alerts(slot, potd_hist, ladder, streaks, slate, t_iso, now)
    except Exception as e:                       # alerts must never break the update
        log(f"alerts failed: {e}")
    save(ROOT / "player_games.json", build_game_logs(idx, slate, now.strftime("%m/%d %I:%M %p ET")))
    log(f"wrote {len(games)} games, {len(props)} props, {len(injuries)} injuries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
