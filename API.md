# DoubleDown AI — API Reference

> For the Lovable frontend at getdoubledown.com. The backend lives at `https://api.getdoubledown.com` (or your configured `VITE_API_URL`).

---

## `GET /api/v1/games` — Upcoming Games Feed (Model-Independent)

Returns every upcoming game with the best available odds across US-regulated sportsbooks. No ML model required — this works with zero data beyond raw odds.

### Response

```json
{
  "count": 25,
  "games": [
    {
      "event_id": "abc123def456",
      "sport": "MLB",
      "sport_key": "baseball_mlb",
      "home_team": "New York Yankees",
      "away_team": "Boston Red Sox",
      "commence_time": "2026-07-12T23:05:00Z",
      "outcomes": [
        {
          "name": "New York Yankees",
          "price": 1.91,
          "best_odds_bookmaker": "FanDuel",
          "consensus_implied_prob": 0.52,
          "all_odds": [
            {"price": 1.91, "bookmaker": "FanDuel"},
            {"price": 1.87, "bookmaker": "DraftKings"},
            {"price": 1.83, "bookmaker": "BetMGM"}
          ]
        },
        {
          "name": "Boston Red Sox",
          "price": 2.1,
          "best_odds_bookmaker": "DraftKings",
          "consensus_implied_prob": 0.48,
          "all_odds": [
            {"price": 2.1, "bookmaker": "DraftKings"},
            {"price": 2.05, "bookmaker": "FanDuel"},
            {"price": 2.0, "bookmaker": "BetRivers"}
          ]
        }
      ]
    }
  ]
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `count` | int | Number of games returned |
| `games[].event_id` | string | Hash from the raw odds event |
| `games[].sport` | string | Human-readable sport name (e.g. "MLB") |
| `games[].sport_key` | string | The Odds API sport key (e.g. "baseball_mlb") |
| `games[].home_team`, `away_team` | string | Team names |
| `games[].commence_time` | ISO 8601 | Game start time (UTC) |
| `games[].outcomes[].name` | string | Team name (or "Draw" for soccer) |
| `games[].outcomes[].price` | float | Best available decimal odds across all books |
| `games[].outcomes[].best_odds_bookmaker` | string|null | Which bookmaker has the best price |
| `games[].outcomes[].consensus_implied_prob` | float|null | Median implied probability across books (0-1) |
| `games[].outcomes[].all_odds` | array | Up to 6 bookmakers sorted by price descending |

### Notes

- Only `h2h` market outcomes are returned
- Odds filtered to 1.10–15.0 (bettable range)
- US-regulated books only: DraftKings, FanDuel, BetMGM, BetRivers, ESPN BET, Bally Bet, Hard Rock Bet, betPARX, Fliff
- Sorted by `commence_time` ascending
- Games before `now` are excluded

---

## `GET /api/v1/health/detailed` — Health Sentinel

Returns the status of all 9 automated health checks. Used by the operator dashboard and the Resend alert system.

### Response

```json
{
  "status": "pass",
  "checks": [
    {
      "name": "odds_freshness",
      "status": "pass",
      "message": "Last fetch: 2026-07-12 12:00:00+00:00 (3.2h ago)"
    },
    {
      "name": "games_board",
      "status": "pass",
      "message": "25 games within next 48 hours"
    },
    {
      "name": "no_past_games",
      "status": "pass",
      "message": "0 games with commence_time in the past"
    },
    {
      "name": "valid_odds",
      "status": "pass",
      "message": "All 150 odds in 1.10-15.0 range from US books"
    },
    {
      "name": "bookmaker_allowlist",
      "status": "pass",
      "message": "All 8 bookmakers in US allowlist"
    },
    {
      "name": "sport_keys",
      "status": "pass",
      "message": "All sport_keys valid"
    },
    {
      "name": "odds_api_quota",
      "status": "warn",
      "message": "142 remaining / 20000 total (0.7%)"
    },
    {
      "name": "db_health",
      "status": "pass",
      "message": "raw_odds: 12865 rows, healthy"
    },
    {
      "name": "grading_pipeline",
      "status": "pass",
      "message": "42 PickOutcomes, last run: 2026-07-12 06:00:00"
    }
  ],
  "summary": {
    "pass": 7,
    "warn": 1,
    "fail": 0,
    "critical": 0
  }
}
```

### Check Reference

| Check | Fails If |
|---|---|
| `odds_freshness` | No fetch in >8 hours |
| `games_board` | 0 games within next 48h |
| `no_past_games` | Any game has past `commence_time` |
| `valid_odds` | Any odds outside 1.10–15.0 |
| `bookmaker_allowlist` | Any bookmaker outside US list |
| `sport_keys` | Unknown sport keys in data |
| `odds_api_quota` | <20% remaining (warn) / <10% (critical) |
| `db_health` | DB unreachable or rows decreasing |
| `grading_pipeline` | No grading run in 48h, or PickOutcomes decreasing |

### Auth
None — public read-only endpoint. Used by the frontend dashboard.

---

## `GET /api/v1/admin/model-progress` — Model Calibration Progress

**Auth required: `X-Admin-Key` header.** Returns the ML model's training progress, calibration stats, and how close we are to the 300-outcome threshold for public model release.

### Request

```
GET /api/v1/admin/model-progress
X-Admin-Key: <your_admin_key>
```

### Response

```json
{
  "graded_outcomes": {
    "total": 42,
    "by_sport": {
      "baseball_mlb": 28,
      "basketball_nba": 14
    },
    "by_day": [
      {"date": "2026-07-11", "count": 22},
      {"date": "2026-07-12", "count": 20}
    ]
  },
  "progress": {
    "threshold": 300,
    "percent_complete": 14.0,
    "daily_rate": 10.0,
    "estimated_threshold_date": "2026-08-01"
  },
  "calibration": [
    {
      "bucket": "50-60%",
      "predicted_avg": null,
      "actual_win_rate": null,
      "sample_size": 0
    },
    {
      "bucket": "60-70%",
      "predicted_avg": null,
      "actual_win_rate": null,
      "sample_size": 0
    },
    {
      "bucket": "70-80%",
      "predicted_avg": null,
      "actual_win_rate": null,
      "sample_size": 0
    },
    {
      "bucket": "80%+",
      "predicted_avg": null,
      "actual_win_rate": null,
      "sample_size": 0
    }
  ],
  "pending": 12,
  "grading_runs": {
    "last_run": "2026-07-12T06:00:00Z",
    "outcomes_graded": 20
  },
  "model": {
    "active_version": "v1.0.0",
    "last_retrained": "2026-07-11T18:00:00Z",
    "real_labels": 42,
    "retrain_status": "ok"
  }
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `graded_outcomes.total` | int | Total PickOutcome records with labels |
| `graded_outcomes.by_sport` | object | Count per sport_key |
| `graded_outcomes.by_day` | array | Daily breakdown (last 30 days) |
| `progress.threshold` | int | 300 — the calibration gate |
| `progress.percent_complete` | float | `total / threshold * 100` |
| `progress.daily_rate` | float | Average labels per day (last 7 days) |
| `progress.estimated_threshold_date` | string | Projected date to hit 300 at current rate |
| `calibration[].bucket` | string | Confidence range (e.g. "50-60%") |
| `calibration[].predicted_avg` | float|null | Mean predicted probability in bucket |
| `calibration[].actual_win_rate` | float|null | Actual win rate in bucket |
| `calibration[].sample_size` | int | Number of outcomes in bucket |
| `pending` | int | Value bets past commence_time awaiting grading |
| `grading_runs.last_run` | ISO 8601 | Last time grading pipeline ran |
| `grading_runs.outcomes_graded` | int | Outcomes processed in last run |
| `model.active_version` | string | Current deployed model version |
| `model.last_retrained` | ISO 8601 | Last retrain timestamp |
| `model.real_labels` | int | Same as `graded_outcomes.total` — live count |
| `model.retrain_status` | string | `ok`, `pending`, or `failed` |

### Error Responses

| Status | Body | When |
|---|---|---|
| 503 | `{"detail": "Admin API key not configured"}` | No `ADMIN_API_KEY` env var set |
| 403 | `{"detail": "Invalid admin key"}` | Wrong/missing `X-Admin-Key` header |

---

## `GET /api/v1/predictions` — Model Picks (Currently Empty)

Returns ML model picks with edge scores. **Currently returns empty results** — the model has not yet accumulated ~300+ graded outcomes for calibration. This endpoint will populate once the calibration threshold is met.

```json
{
  "count": 0,
  "predictions": []
}
```

The Double Down ($39/mo) subscription tier unlocks access to this data when it becomes available.