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

SEASON = "2026-27"
SEASON_START = date(2026, 10, 20)
ET = ZoneInfo("America/New_York")
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ODDS = "https://api.the-odds-api.com/v4/sports/basketball_nba"
ODDS_KEY = os.environ.get("ODDS_API_KEY", "").strip()
MAX_PROP_EVENTS = int(os.environ.get("MAX_PROP_EVENTS", "30"))

PROP_MARKETS = {
    "player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST",
    "player_threes": "3PM", "player_blocks": "BLK", "player_steals": "STL",
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


def project_player(games, opp_factor):
    """games: chronological list of box-score rows. Returns {stat: projection} or None."""
    recent = [g for g in games if g["min"] >= 5][-10:][::-1]   # newest first
    if len(recent) < 3:
        return None
    mins = wmean([g["min"] for g in recent[:8]], 0.85)
    if mins < 12:
        return None
    tot_min = sum(g["min"] for g in recent)
    rate = {k: sum(g[k] for g in recent) / tot_min for k in ("pts", "reb", "ast", "tpm", "blk", "stl")}
    p = {"PTS": rate["pts"] * mins * opp_factor, "REB": rate["reb"] * mins, "AST": rate["ast"] * mins,
         "3PM": rate["tpm"] * mins, "BLK": rate["blk"] * mins, "STL": rate["stl"] * mins}
    p["PRA"] = p["PTS"] + p["REB"] + p["AST"]
    p["PR"] = p["PTS"] + p["REB"]
    p["PA"] = p["PTS"] + p["AST"]
    p["RA"] = p["REB"] + p["AST"]
    return {k: rnd(v) for k, v in p.items()}


def odds_props(slate, plog_idx):
    """DraftKings player-prop lines via the-odds-api.com. Returns {event_id: [(player, stat, line)]}."""
    if not ODDS_KEY:
        return {}
    evs = get(f"{ODDS}/events", apiKey=ODDS_KEY)
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
        j = get(f"{ODDS}/events/{oid}/odds", apiKey=ODDS_KEY, regions="us", bookmakers="draftkings",
                markets=",".join(PROP_MARKETS), oddsFormat="american")
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
    j = get(f"{ODDS}/odds", apiKey=ODDS_KEY, regions="us", bookmakers="draftkings",
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
                elif m["key"] == "totals" and "Over" in oc and oc["Over"].get("point") is not None:
                    row["total"] = float(oc["Over"]["point"])
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
    j = get("https://api.the-odds-api.com/v4/sports/basketball_nba_championship_winner/odds", apiKey=ODDS_KEY,
            regions="us", markets="outrights", oddsFormat="american")
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
    idx = player_index(plog)
    injuries = fetch_injuries(slate)
    if injuries is None:
        injuries = prev.get("injuries", [])
    out_set = {(norm(i["player"])) for i in injuries if i["status"].lower() == "out"}
    upcoming = [g for g in slate if g["status"] == "Upcoming"]
    prop_lines = odds_props(upcoming, idx) if upcoming else {}
    gml = odds_game_lines() if upcoming else {}
    for g in upcoming:
        hit = gml.get((NAME_FIX.get(g["names"][g["away"]], g["names"][g["away"]]), NAME_FIX.get(g["names"][g["home"]], g["names"][g["home"]])))
        if hit:
            g["_lines"] = {**hit, "provider": "DraftKings"}
    rest = team_rest(plog, slate)
    standings = fetch_standings() or prev.get("standings")
    futures = fetch_futures() or prev.get("futures")

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
                of = 1.0
                if opp in ratings and lg:
                    of = min(1.05, max(0.95, ratings[opp]["def"] / lg))
                proj = project_player(hist, of)
                if not proj or stat not in proj:
                    continue
                fresh.append({"player": hist[-1]["n"], "team": team, "game": g["id"], "stat": stat,
                              "model": proj[stat], "dk": line, "over": over_px, "under": under_px})
            if fresh:
                pending["props"][eid] = fresh
        for p in pending["props"].get(eid, []):
            e = rnd(p["model"] - p["dk"])
            props.append({**p, "edge": e, "side": "Over" if e > 0 else "Under"})

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
                            "Adjustments: opponent defense (capped +/-5%), home court for games; injured players skipped",
                            f"Residual layer: alpha fades from 0.45 toward 0.12 as games are learned (now {state['residuals']['alpha']})",
                            "Benchmark: DraftKings via The Odds API (games and props), ESPN as backup",
                            "Bets are scored at 1 unit risked at the posted DraftKings price (-110 when no price was posted)"],
            "prediction_history": state["prediction_history"], "notes": note,
            "settled_bets": state["settled_bets"], "record": state["record"],
            "accuracy": state["accuracy"], "standings": standings,
            "futures": {"title": futures, "book": futures[0].get("book", ""), "updated": now.strftime("%m/%d %I:%M %p ET")} if isinstance(futures, list) else futures}

    # validate before replacing anything
    json.dumps(data)
    assert isinstance(data["games"], list) and isinstance(data["props"], list)
    save(LOG_PATH, plog)
    save(PENDING_PATH, pending)
    save(OUT, data, indent=1)
    save(ROOT / "player_games.json", build_game_logs(idx, slate, now.strftime("%m/%d %I:%M %p ET")))
    log(f"wrote {len(games)} games, {len(props)} props, {len(injuries)} injuries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
