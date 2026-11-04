# NBA PJ Model

Static site on GitHub Pages. `index.html` loads `projections_data.json` when opened.

## How it runs itself
`.github/workflows/update.yml` runs `update_projections.py` at 1:00 AM and 11:00 AM ET every day (DST-safe) and on demand from the Actions tab.
The script:
1. pulls finished box scores from ESPN into `data/player_log.json`,
2. scores yesterday's frozen predictions (`data/pending.json`), updating residuals, record and settled bets,
3. projects the next slate, and writes `projections_data.json` and `player_games.json` (last 15 games per rotation player, used by the Game Log tab).
The workflow commits those files back to the repo, and Pages republishes.

## Setup
- Player props need a key from the-odds-api.com. Add it as a repo secret named `ODDS_API_KEY`
  (Settings > Secrets and variables > Actions). Without it, game lines (from ESPN) still update and props stay empty.
- First run backfills box scores from the season start, so it takes a few minutes.
- Injuries are fetched live from ESPN by the page itself (on open and every 5 min), so they are current between runs.
- GitHub may start scheduled runs a few minutes late; that is normal.

## Old games
Only live, today's and upcoming games are published. Finished games are kept in `data/` for scoring only.
The page also hides any game dated before today (ET) and shows a banner if the data is over 14 hours old.
