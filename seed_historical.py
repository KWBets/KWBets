"""Seed historical game outcomes + futures market data for ML training.

Generates 1-2 seasons of realistic historical sports data so the XGBoost
model can train on labeled outcomes before surfacing live value bets.

Covers:
- In-season: MLB, Tennis (Wimbledon), Golf (US Open), Soccer (World Cup)
- Futures: NFL Super Bowl, NBA Championship, NHL Stanley Cup, Division winners, MVP, win totals
"""
import random
import uuid
from datetime import datetime, timedelta, timezone

from app.database import SessionLocal, init_db
from app.models import (
    ProcessedFeatures, ModelPrediction, ModelRegistry,
    ValueBet, PickOutcome, RawOdds,
)


random.seed(42)

# ---------------------------------------------------------------------------
# Sports / leagues with realistic team data
# ---------------------------------------------------------------------------

SEASONS = {
    "nfl": {"sport_key": "americanfootball_nfl", "sport": "NFL", "type": "futures", "games": 0},
    "nba": {"sport_key": "basketball_nba", "sport": "NBA", "type": "futures", "games": 0},
    "nhl": {"sport_key": "icehockey_nhl", "sport": "NHL", "type": "futures", "games": 0},
    "mlb": {"sport_key": "baseball_mlb", "sport": "MLB", "type": "inseason", "games": 162},
    "worldcup": {"sport_key": "soccer_fifa_world_cup", "sport": "World Cup", "type": "inseason", "games": 64},
    "tennis": {"sport_key": "tennis_atp_wimbledon", "sport": "Wimbledon", "type": "inseason", "games": 127},
    "golf": {"sport_key": "golf_us_open", "sport": "US Open Golf", "type": "inseason", "games": 0},
}

# ---------------------------------------------------------------------------
# In-season team matchups (for h2h games)
# ---------------------------------------------------------------------------

MLB_TEAMS = [
    "Yankees", "Red Sox", "Dodgers", "Braves", "Astros", "Phillies",
    "Orioles", "Rays", "Padres", "Cubs", "Brewers", "Twins",
    "Rangers", "Mariners", "Blue Jays", "Guardians",
]

TENNIS_PLAYERS = [
    "Carlos Alcaraz", "Jannik Sinner", "Novak Djokovic", "Alexander Zverev",
    "Daniil Medvedev", "Taylor Fritz", "Stefanos Tsitsipas", "Andrey Rublev",
    "Casper Ruud", "Hubert Hurkacz", "Alex de Minaur", "Grigor Dimitrov",
    "Ben Shelton", "Frances Tiafoe", "Tommy Paul", "Jack Draper",
]

GOLF_PLAYERS = [
    "Scottie Scheffler", "Rory McIlroy", "Xander Schauffele", "Jon Rahm",
    "Viktor Hovland", "Collin Morikawa", "Patrick Cantlay", "Jordan Spieth",
    "Ludvig Aberg", "Brooks Koepka", "Bryson DeChambeau", "Max Homa",
    "Wyndham Clark", "Matt Fitzpatrick", "Tommy Fleetwood", "Hideki Matsuyama",
]

WORLD_CUP_TEAMS = [
    ("Argentina", "France"), ("Brazil", "England"), ("Spain", "Germany"),
    ("Portugal", "Netherlands"), ("Croatia", "Morocco"), ("Belgium", "Italy"),
    ("Uruguay", "Denmark"), ("Switzerland", "Japan"),
]

# ---------------------------------------------------------------------------
# Futures markets (offseason)
# ---------------------------------------------------------------------------

NFL_TEAMS = [
    "Chiefs", "49ers", "Eagles", "Ravens", "Bengals", "Bills", "Cowboys",
    "Lions", "Packers", "Dolphins", "Texans", "Jets", "Chargers", "Bears",
    "Vikings", "Falcons", "Saints", "Steelers", "Browns", "Raiders",
    "Seahawks", "Giants", "Commanders", "Rams", "Buccaneers", "Jaguars",
    "Colts", "Broncos", "Titans", "Patriots", "Panthers", "Cardinals",
]

NBA_TEAMS = [
    "Celtics", "Lakers", "Warriors", "Nuggets", "Bucks", "Thunder",
    "Mavericks", "Timberwolves", "Knicks", "Sixers", "Heat", "Suns",
    "Clippers", "Pelicans", "Pacers", "Cavaliers", "Magic", "Kings",
    "Rockets", "Hawks", "Bulls", "Jazz", "Grizzlies", "Raptors",
    "Spurs", "Nets", "Blazers", "Wizards", "Hornets", "Pistons",
]

NHL_TEAMS = [
    "Panthers", "Oilers", "Avalanche", "Stars", "Bruins", "Maple Leafs",
    "Canucks", "Rangers", "Hurricanes", "Jets", "Penguins", "Kings",
    "Lightning", "Devils", "Predators", "Islanders", "Capitals", "Kraken",
    "Sabres", "Senators", "Flames", "Canadiens", "Ducks", "Sharks",
    "Blue Jackets", "Coyotes", "Red Wings", "Wild", "Blackhawks", "Blues",
]

FUTURES_MARKETS = {
    "nfl": [
        ("Super Bowl Winner", 1),
        ("AFC Winner", 2),
        ("NFC Winner", 2),
        ("Division Winner - AFC East", 4),
        ("Division Winner - AFC North", 4),
        ("Division Winner - AFC South", 4),
        ("Division Winner - AFC West", 4),
        ("Division Winner - NFC East", 4),
        ("Division Winner - NFC North", 4),
        ("Division Winner - NFC South", 4),
        ("Division Winner - NFC West", 4),
        ("NFL MVP", 1),
        ("Offensive Player of the Year", 1),
        ("Defensive Player of the Year", 1),
        ("Offensive Rookie of the Year", 1),
        ("Defensive Rookie of the Year", 1),
    ],
    "nba": [
        ("NBA Championship Winner", 1),
        ("Eastern Conference Winner", 1),
        ("Western Conference Winner", 1),
        ("Division Winner - Atlantic", 5),
        ("Division Winner - Central", 5),
        ("Division Winner - Southeast", 5),
        ("Division Winner - Northwest", 5),
        ("Division Winner - Pacific", 5),
        ("Division Winner - Southwest", 5),
        ("NBA MVP", 1),
        ("Rookie of the Year", 1),
        ("Defensive Player of the Year", 1),
        ("Sixth Man of the Year", 1),
        ("Most Improved Player", 1),
    ],
    "nhl": [
        ("Stanley Cup Winner", 1),
        ("Eastern Conference Winner", 1),
        ("Western Conference Winner", 1),
        ("Division Winner - Atlantic", 8),
        ("Division Winner - Metropolitan", 8),
        ("Division Winner - Central", 8),
        ("Division Winner - Pacific", 8),
        ("Hart Memorial Trophy (MVP)", 1),
        ("Vezina Trophy (Best Goalie)", 1),
        ("Norris Trophy (Best Defenseman)", 1),
        ("Calder Trophy (Best Rookie)", 1),
        ("Conn Smythe Trophy (Playoff MVP)", 1),
    ],
}

LEAGUE_TEAMS = {"nfl": NFL_TEAMS, "nba": NBA_TEAMS, "nhl": NHL_TEAMS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_odds(min_odds=1.1, max_odds=15.0):
    """Generate random decimal odds following a realistic distribution."""
    # Most odds cluster in the 1.5-5.0 range with a long tail
    r = random.random()
    if r < 0.3:
        return round(random.uniform(1.1, 1.8), 2)
    elif r < 0.6:
        return round(random.uniform(1.8, 3.0), 2)
    elif r < 0.85:
        return round(random.uniform(3.0, 6.0), 2)
    else:
        return round(random.uniform(6.0, max_odds), 2)


def implied_prob_from_odds(odds):
    return 1.0 / odds


def simulate_outcome(home_implied_prob, home_team, away_team):
    """Simulate a game outcome based on implied probability."""
    # Add some noise — the market isn't perfectly efficient
    noisy_prob = home_implied_prob + random.uniform(-0.08, 0.08)
    noisy_prob = max(0.1, min(0.9, noisy_prob))

    if random.random() < noisy_prob:
        return "won", 1, home_team
    else:
        return "won", 0, away_team  # away winner


def simulate_futures_outcome(team_implied_prob, team_name):
    """Simulate whether a futures bet hits."""
    noisy_prob = team_implied_prob + random.uniform(-0.05, 0.05)
    noisy_prob = max(0.01, min(0.5, noisy_prob))
    return "won" if random.random() < noisy_prob else "lost"


# ---------------------------------------------------------------------------
# Seed in-season h2h games
# ---------------------------------------------------------------------------

def seed_inseason_games(db, num_games=300):
    """Seed h2h games for MLB, Tennis, Golf, World Cup."""
    print("[seed_historical] Seeding in-season h2h games...")
    created = 0

    for _ in range(num_games):
        # Pick a random sport
        league = random.choice(["mlb", "worldcup", "tennis", "golf"])
        league_info = SEASONS[league]
        now = datetime.now(timezone.utc)

        # Game date: 30-400 days in the past
        days_ago = random.randint(30, 400)
        game_date = now - timedelta(days=days_ago, hours=random.randint(0, 23))
        event_id = f"hist_{league}_{uuid.uuid4().hex[:8]}"

        if league == "tennis":
            home_team = random.choice(TENNIS_PLAYERS)
            away_team = random.choice([p for p in TENNIS_PLAYERS if p != home_team])
        elif league == "golf":
            # Golf is individual — model as h2h matchups in tournament
            home_team = random.choice(GOLF_PLAYERS)
            away_team = random.choice([p for p in GOLF_PLAYERS if p != home_team])
        elif league == "worldcup":
            home_team, away_team = random.choice(WORLD_CUP_TEAMS)
        else:
            home_team = random.choice(MLB_TEAMS)
            away_team = random.choice([t for t in MLB_TEAMS if t != home_team])

        home_odds = random_odds()
        away_odds = random_odds()

        home_implied = implied_prob_from_odds(home_odds)
        away_implied = implied_prob_from_odds(away_odds)

        # Remove juice
        total = home_implied + away_implied
        home_true = home_implied / total
        away_true = away_implied / total

        # Create ProcessedFeatures
        odds_displayed = random.randint(2, 6)
        home_odds_std = round(random.uniform(0.05, 0.4), 3)

        pf = ProcessedFeatures(
            sport=league_info["sport"],
            sport_key=league_info["sport_key"],
            event_id=event_id,
            home_team=home_team,
            away_team=away_team,
            commence_time=game_date,
            market_type="h2h",
            home_implied_prob=round(home_true, 4),
            away_implied_prob=round(away_true, 4),
            draw_implied_prob=None,
            home_odds_avg=home_odds,
            away_odds_avg=away_odds,
            draw_odds_avg=None,
            home_spread=None,
            away_spread=None,
            over_line=None,
            under_line=None,
            odds_displayed_count=odds_displayed,
            home_odds_std=home_odds_std,
            away_odds_std=round(random.uniform(0.05, 0.4), 3),
            created_at=game_date,
        )
        db.add(pf)
        db.flush()

        # Create ValueBet — model has slight edge over market
        # Simulate model being right ~55% of the time
        if random.random() < 0.55:
            team = home_team
            model_prob = min(home_true + random.uniform(0.02, 0.10), 0.95)
            implied_prob = home_true
            odds_val = home_odds
        else:
            team = away_team
            model_prob = min(away_true + random.uniform(0.02, 0.10), 0.95)
            implied_prob = away_true
            odds_val = away_odds

        edge = (model_prob - implied_prob) * 100
        ev = (model_prob * odds_val) - 1

        if edge < 1.0:
            # Skip bets with no meaningful edge
            continue

        confidence = "high" if edge > 8 else "medium" if edge > 4 else "low"
        model_ver = f"train-v{random.randint(2024, 2026)}"

        vb = ValueBet(
            event_id=event_id,
            sport=league_info["sport"],
            sport_key=league_info["sport_key"],
            home_team=home_team,
            away_team=away_team,
            commence_time=game_date,
            team=team,
            market_type="h2h",
            pick_label=f"{team} ML",
            odds=odds_val,
            model_probability=round(model_prob, 4),
            implied_probability=round(implied_prob, 4),
            edge_percentage=round(edge, 2),
            expected_value=round(ev, 4),
            confidence_tier=confidence,
            confidence_score=round(min(abs(edge) / 15.0, 1.0), 2),
            reasoning_factors={
                "form": round(random.uniform(-0.03, 0.05), 3),
                "rest_days": round(random.uniform(-0.02, 0.03), 3),
                "market_movement": round(random.uniform(-5, 5), 1),
            },
            model_version=model_ver,
            best_bookmaker=random.choice(["draftkings", "fanduel", "betmgm"]),
            status="pending",
            created_at=game_date,
            updated_at=game_date,
        )
        db.add(vb)
        db.flush()

        # Create PickOutcome — simulate actual result
        actual_outcome, result_val, winner = simulate_outcome(home_true, home_team, away_team)

        if team == winner:
            pick_result = "won"
        else:
            pick_result = "lost"

        po = PickOutcome(
            value_bet_id=vb.id,
            event_id=event_id,
            actual_outcome=actual_outcome,
            home_score=random.randint(0, 10),
            away_score=random.randint(0, 10),
            resolved_at=game_date + timedelta(hours=3, minutes=random.randint(0, 120)),
            created_at=game_date + timedelta(hours=3),
        )
        db.add(po)
        created += 1

        # Create ModelPrediction for completeness
        mp = ModelPrediction(
            event_id=event_id,
            sport=league_info["sport"],
            home_team=home_team,
            away_team=away_team,
            commence_time=game_date,
            market_type="h2h",
            home_win_probability=round(home_true + random.uniform(-0.03, 0.03), 4),
            away_win_probability=round(away_true + random.uniform(-0.03, 0.03), 4),
            model_version=model_ver,
            model_run_id=uuid.uuid4().hex,
            feature_timestamp=game_date - timedelta(hours=random.randint(1, 24)),
            created_at=game_date,
        )
        db.add(mp)

    db.commit()
    print(f"[seed_historical] Created {created} in-season h2h game records.")
    return created


# ---------------------------------------------------------------------------
# Seed futures markets (offseason)
# ---------------------------------------------------------------------------

def seed_futures_markets(db, num_entries=200):
    """Seed futures market data for NFL, NBA, NHL."""
    print("[seed_historical] Seeding futures markets...")
    created = 0
    now = datetime.now(timezone.utc)

    for league_key, markets in FUTURES_MARKETS.items():
        league_info = SEASONS[league_key]
        teams = LEAGUE_TEAMS[league_key]

        for market_name, num_outcomes in markets:
            # Pick candidates for this market
            if num_outcomes <= 1:
                candidates = random.sample(teams, min(12, len(teams)))
            else:
                # Division/conference markets — pick teams that would be in that group
                candidates = random.sample(teams, min(num_outcomes + 5, len(teams)))

            for candidate in candidates:
                event_id = f"futures_{league_key}_{market_name[:10]}_{candidate[:5]}_{uuid.uuid4().hex[:6]}"

                # Futures have longer time horizons — expiry 150-365 days from now
                start_date = now - timedelta(days=random.randint(30, 200))
                expiry_date = start_date + timedelta(days=random.randint(150, 365))
                resolution_date = expiry_date + timedelta(days=random.randint(0, 30))

                # Futures odds are typically longer
                odds = random_odds(1.5, 40.0)
                implied = implied_prob_from_odds(odds)

                # Create ProcessedFeatures for this futures market entry
                pf = ProcessedFeatures(
                    sport=league_info["sport"],
                    sport_key=league_info["sport_key"],
                    event_id=event_id,
                    home_team=candidate,
                    away_team=f"Field ({market_name})",
                    commence_time=resolution_date,
                    market_type="h2h",
                    home_implied_prob=round(implied, 4),
                    away_implied_prob=round(1.0 - implied, 4) if implied < 1.0 else 0.01,
                    draw_implied_prob=None,
                    home_odds_avg=odds,
                    away_odds_avg=round(1.0 / (1.0 - implied), 2) if implied < 1.0 else None,
                    draw_odds_avg=None,
                    odds_displayed_count=random.randint(1, 4),
                    home_odds_std=round(random.uniform(0.1, 0.8), 3),
                    away_odds_std=round(random.uniform(0.1, 0.8), 3),
                    created_at=start_date,
                )
                db.add(pf)
                db.flush()

                # ValueBet — model thinks this candidate is undervalued
                model_prob = min(implied + random.uniform(0.01, 0.08), 0.4)
                edge = (model_prob - implied) * 100
                ev = (model_prob * odds) - 1

                if edge < 1.0:
                    continue

                confidence = "medium" if edge > 5 else "low"
                model_ver = f"futures-v{random.randint(2024, 2026)}"

                vb = ValueBet(
                    event_id=event_id,
                    sport=league_info["sport"],
                    sport_key=league_info["sport_key"],
                    home_team=candidate,
                    away_team=f"Field ({market_name})",
                    commence_time=resolution_date,
                    team=candidate,
                    market_type="h2h",
                    pick_label=f"{candidate} — {market_name}",
                    odds=odds,
                    model_probability=round(model_prob, 4),
                    implied_probability=round(implied, 4),
                    edge_percentage=round(edge, 2),
                    expected_value=round(ev, 4),
                    confidence_tier=confidence,
                    confidence_score=round(min(abs(edge) / 15.0, 1.0), 2),
                    reasoning_factors={
                        "market_type": "futures",
                        "time_horizon_days": (resolution_date - start_date).days,
                        "team_strength": round(random.uniform(0.0, 1.0), 2),
                        "offseason_changes": round(random.uniform(-0.03, 0.05), 3),
                    },
                    model_version=model_ver,
                    best_bookmaker=random.choice(["draftkings", "fanduel", "betmgm", "caesars"]),
                    status="pending",
                    created_at=start_date,
                    updated_at=start_date,
                )
                db.add(vb)
                db.flush()

                # PickOutcome — simulate if the futures bet hit
                pick_result = simulate_futures_outcome(implied, candidate)
                if pick_result == "won":
                    actual = "won"
                else:
                    actual = "lost"

                po = PickOutcome(
                    value_bet_id=vb.id,
                    event_id=event_id,
                    actual_outcome=actual,
                    resolved_at=resolution_date,
                    created_at=resolution_date,
                )
                db.add(po)

                # ModelPrediction
                mp = ModelPrediction(
                    event_id=event_id,
                    sport=league_info["sport"],
                    home_team=candidate,
                    away_team=f"Field ({market_name})",
                    commence_time=resolution_date,
                    market_type="h2h",
                    home_win_probability=round(model_prob, 4),
                    away_win_probability=round(1.0 - model_prob, 4),
                    model_version=model_ver,
                    model_run_id=uuid.uuid4().hex,
                    feature_timestamp=start_date,
                    created_at=start_date,
                )
                db.add(mp)
                created += 1

    db.commit()
    print(f"[seed_historical] Created {created} futures market records.")
    return created


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  DoubleDown AI — Historical Data Seeder")
    print("=" * 60)

    # Clear existing seed data
    init_db()
    db = SessionLocal()

    try:
        # Don't clear raw_odds — those are live from the API
        db.query(PickOutcome).delete()
        db.query(ValueBet).delete()
        db.query(ModelPrediction).delete()
        db.query(ModelRegistry).delete()
        db.query(ProcessedFeatures).delete()
        db.commit()

        inseason = seed_inseason_games(db, num_games=350)
        futures = seed_futures_markets(db, num_entries=250)

        print(f"\n[seed_historical] Total created: {inseason + futures} records")
        print(f"  - In-season h2h games:  {inseason}")
        print(f"  - Futures market entries: {futures}")

        # Final counts
        from sqlalchemy import func
        for table in [ProcessedFeatures, ValueBet, PickOutcome, ModelPrediction, ModelRegistry]:
            count = db.query(func.count(table.id)).scalar()
            print(f"  - {table.__tablename__}: {count}")

    finally:
        db.close()


if __name__ == "__main__":
    main()