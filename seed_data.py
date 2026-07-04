"""Seed the database with realistic sample data for development and testing."""

import random
from datetime import datetime, timedelta, timezone

from app.database import SessionLocal, init_db
from app.models import (
    RawOdds,
    ProcessedFeatures,
    ModelPrediction,
    ValueBet,
    ModelRegistry,
)


# ---------------------------------------------------------------------------
# Sample sports data
# ---------------------------------------------------------------------------
SPORTS_DATA = {
    "americanfootball_nfl": "NFL",
    "basketball_nba": "NBA",
    "baseball_mlb": "MLB",
    "icehockey_nhl": "NHL",
    "soccer_epl": "English Premier League",
}

TEAMS = {
    "americanfootball_nfl": [
        ("Kansas City Chiefs", "San Francisco 49ers"),
        ("Baltimore Ravens", "Cincinnati Bengals"),
        ("Philadelphia Eagles", "Dallas Cowboys"),
        ("Detroit Lions", "Green Bay Packers"),
        ("Buffalo Bills", "Miami Dolphins"),
    ],
    "basketball_nba": [
        ("Boston Celtics", "Los Angeles Lakers"),
        ("Denver Nuggets", "Golden State Warriors"),
        ("Milwaukee Bucks", "Miami Heat"),
        ("Oklahoma City Thunder", "Dallas Mavericks"),
        ("New York Knicks", "Philadelphia 76ers"),
    ],
    "baseball_mlb": [
        ("Los Angeles Dodgers", "New York Yankees"),
        ("Atlanta Braves", "Philadelphia Phillies"),
        ("Houston Astros", "Texas Rangers"),
        ("Baltimore Orioles", "Tampa Bay Rays"),
        ("San Diego Padres", "Chicago Cubs"),
    ],
    "icehockey_nhl": [
        ("Florida Panthers", "Edmonton Oilers"),
        ("Colorado Avalanche", "Dallas Stars"),
        ("Boston Bruins", "Toronto Maple Leafs"),
        ("Vancouver Canucks", "Edmonton Oilers"),
        ("Carolina Hurricanes", "New York Rangers"),
    ],
    "soccer_epl": [
        ("Manchester City", "Arsenal"),
        ("Liverpool", "Chelsea"),
        ("Manchester United", "Tottenham Hotspur"),
        ("Newcastle United", "Aston Villa"),
        ("Brighton", "West Ham United"),
    ],
}


def seed_value_bets(db):
    """Populate the value_bets table with realistic sample data."""
    print("[seed] Creating sample value bets...")

    now = datetime.now(timezone.utc)
    count = 0

    for sport_key, sport_title in SPORTS_DATA.items():
        matchups = TEAMS.get(sport_key, [])

        for home_team, away_team in matchups:
            # Randomize commence time: 1-7 days from now
            days_out = random.randint(1, 7)
            commence_time = now + timedelta(days=days_out, hours=random.randint(0, 23))

            # Simulate bookmaker odds (decimal)
            home_odds = round(random.uniform(1.5, 3.5), 2)
            away_odds = round(random.uniform(1.5, 3.5), 2)

            # Implied probabilities (with bookmaker juice ~4%)
            juice = 1.04
            home_implied = 1.0 / home_odds * juice
            away_implied = 1.0 / away_odds * juice

            # Normalize to remove juice for "true" implied
            total = home_implied + away_implied
            home_true_implied = home_implied / total
            away_true_implied = away_implied / total

            # Model probabilities — slightly better than implied for the value side
            # Home team gets an edge ~60% of the time (randomize)
            if random.random() < 0.6:
                # Home team is the value bet
                model_prob = home_true_implied + random.uniform(0.02, 0.12)
                edge = (model_prob - home_true_implied) * 100
                team = home_team
                pick_label = f"{home_team} (Moneyline)"
                implied_prob = home_true_implied
                odds_val = home_odds
                confidence = "high" if edge > 8 else "medium" if edge > 4 else "low"
                factors = {
                    "home_advantage": round(random.uniform(0.02, 0.05), 3),
                    "recent_form": round(random.uniform(0.01, 0.04), 3),
                    "injury_impact": round(random.uniform(-0.02, 0.03), 3),
                    "rest_days": round(random.uniform(0.0, 0.02), 3),
                }
            else:
                # Away team is the value bet
                model_prob = away_true_implied + random.uniform(0.02, 0.12)
                edge = (model_prob - away_true_implied) * 100
                team = away_team
                pick_label = f"{away_team} (Moneyline)"
                implied_prob = away_true_implied
                odds_val = away_odds
                confidence = "high" if edge > 8 else "medium" if edge > 4 else "low"
                factors = {
                    "home_advantage": round(random.uniform(-0.03, 0.0), 3),
                    "recent_form": round(random.uniform(0.02, 0.05), 3),
                    "injury_impact": round(random.uniform(-0.01, 0.02), 3),
                    "rest_days": round(random.uniform(0.0, 0.02), 3),
                }

            ev = round((model_prob * odds_val) - 1, 4)

            event_id = f"seed_{sport_key}_{home_team[:3]}_{away_team[:3]}_{count}"

            value_bet = ValueBet(
                event_id=event_id,
                sport=sport_title,
                sport_key=sport_key,
                home_team=home_team,
                away_team=away_team,
                commence_time=commence_time,
                team=team,
                market_type="h2h",
                pick_label=pick_label,
                odds=odds_val,
                model_probability=round(model_prob, 4),
                implied_probability=round(implied_prob, 4),
                edge_percentage=round(edge, 2),
                expected_value=round(ev, 4),
                confidence_tier=confidence,
                confidence_score=round(abs(edge) / 15.0, 2) if edge > 0 else 0.1,
                reasoning_factors=factors,
                model_version="seed-v1.0",
                best_bookmaker=random.choice([
                    "draftkings", "fanduel", "betmgm", "caesars", "pointsbetus"
                ]),
                status="pending",
                created_at=now,
                updated_at=now,
            )
            db.add(value_bet)
            count += 1

            # Also create a model prediction for this event
            pred = ModelPrediction(
                event_id=event_id,
                sport=sport_title,
                home_team=home_team,
                away_team=away_team,
                commence_time=commence_time,
                market_type="h2h",
                home_win_probability=round(model_prob if team == home_team else (1 - model_prob), 4),
                away_win_probability=round(model_prob if team == away_team else (1 - model_prob), 4),
                model_version="seed-v1.0",
                model_run_id="seed-run-001",
                feature_timestamp=now - timedelta(hours=random.randint(1, 12)),
                created_at=now,
            )
            db.add(pred)

    db.commit()
    print(f"[seed] Created {count} sample value bets and {count} model predictions.")
    return count


def seed_raw_odds(db):
    """Create sample raw odds records so the odds endpoints have data."""
    from app.models import RawOdds

    print("[seed] Creating sample raw odds...")
    now = datetime.now(timezone.utc)
    count = 0

    bookmakers = [
        ("draftkings", "DraftKings"),
        ("fanduel", "FanDuel"),
        ("betmgm", "BetMGM"),
        ("caesars", "Caesars"),
        ("pointsbetus", "PointsBet"),
    ]

    for sport_key, sport_title in SPORTS_DATA.items():
        matchups = TEAMS.get(sport_key, [])
        for home_team, away_team in matchups:
            days_out = random.randint(1, 7)
            commence_time = now + timedelta(days=days_out, hours=random.randint(0, 23))
            event_id = f"seed_{sport_key}_{home_team[:3]}_{away_team[:3]}_{count}"

            base_home_odds = round(random.uniform(1.5, 3.5), 2)
            base_away_odds = round(random.uniform(1.5, 3.5), 2)

            for bm_key, bm_title in bookmakers:
                # Slight variation per bookmaker
                home_odds = round(base_home_odds + random.uniform(-0.15, 0.15), 2)
                away_odds = round(base_away_odds + random.uniform(-0.15, 0.15), 2)

                # H2H market
                for outcome_name, price in [(home_team, home_odds), (away_team, away_odds)]:
                    row_id = f"{event_id}_{bm_key}_h2h_{outcome_name}"
                    row = RawOdds(
                        id=row_id,
                        sport=sport_title,
                        sport_key=sport_key,
                        commence_time=commence_time,
                        home_team=home_team,
                        away_team=away_team,
                        bookmaker_key=bm_key,
                        bookmaker_title=bm_title,
                        market_key="h2h",
                        outcome_name=outcome_name,
                        outcome_price=price,
                        odds_timestamp=now - timedelta(hours=random.randint(0, 4)),
                        fetched_at=now,
                        last_update=now,
                    )
                    db.add(row)
                    count += 1

            # Spreads market
            home_spread = round(random.uniform(-7.5, -1.5), 1)
            away_spread = round(abs(home_spread), 1)
            for outcome_name, price, point in [
                (home_team, round(random.uniform(1.8, 2.1), 2), home_spread),
                (away_team, round(random.uniform(1.8, 2.1), 2), away_spread),
            ]:
                row_id = f"{event_id}_fanduel_spreads_{outcome_name}"
                row = RawOdds(
                    id=row_id,
                    sport=sport_title,
                    sport_key=sport_key,
                    commence_time=commence_time,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_key="fanduel",
                    bookmaker_title="FanDuel",
                    market_key="spreads",
                    outcome_name=outcome_name,
                    outcome_price=price,
                    outcome_point=point,
                    odds_timestamp=now - timedelta(hours=random.randint(0, 4)),
                    fetched_at=now,
                    last_update=now,
                )
                db.add(row)
                count += 1

            # Totals market
            total_line = round(random.uniform(195.0, 230.0) if "basketball" in sport_key else random.uniform(40.0, 55.0), 1)
            for outcome_name, price in [
                (f"Over {total_line}", round(random.uniform(1.85, 2.0), 2)),
                (f"Under {total_line}", round(random.uniform(1.85, 2.0), 2)),
            ]:
                row_id = f"{event_id}_draftkings_totals_{outcome_name.replace(' ', '_')}"
                row = RawOdds(
                    id=row_id,
                    sport=sport_title,
                    sport_key=sport_key,
                    commence_time=commence_time,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_key="draftkings",
                    bookmaker_title="DraftKings",
                    market_key="totals",
                    outcome_name=outcome_name,
                    outcome_price=price,
                    outcome_point=total_line,
                    odds_timestamp=now - timedelta(hours=random.randint(0, 4)),
                    fetched_at=now,
                    last_update=now,
                )
                db.add(row)
                count += 1

    db.commit()
    print(f"[seed] Created {count} sample raw odds rows.")
    return count


def register_seed_model(db):
    """Register the seed model in the model registry."""
    model_entry = ModelRegistry(
        model_name="xgboost_classifier",
        model_version="seed-v1.0",
        model_type="xgboost",
        model_path="models/saved/seed_model.pkl",
        feature_set_version="v1.0",
        roc_auc=0.72,
        accuracy=0.68,
        precision=0.65,
        recall=0.70,
        f1_score=0.67,
        log_loss=0.58,
        brier_score=0.22,
        training_start=datetime.now(timezone.utc) - timedelta(days=1),
        training_end=datetime.now(timezone.utc) - timedelta(hours=12),
        training_samples=12500,
        hyperparameters={
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
        },
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(model_entry)
    db.commit()
    print("[seed] Registered seed model v1.0 in model registry.")


def main():
    print("=" * 60)
    print("  DoubleDown AI — Seed Data Generator")
    print("=" * 60)

    init_db()
    db = SessionLocal()

    try:
        # Clear existing seed data to avoid duplicates
        db.query(ValueBet).delete()
        db.query(ModelPrediction).delete()
        db.query(RawOdds).delete()
        db.query(ProcessedFeatures).delete()
        db.commit()

        seed_value_bets(db)
        seed_raw_odds(db)
        register_seed_model(db)

        # Quick counts
        counts = {
            "value_bets": db.query(ValueBet).count(),
            "model_predictions": db.query(ModelPrediction).count(),
            "raw_odds": db.query(RawOdds).count(),
            "model_registry": db.query(ModelRegistry).count(),
        }
        print(f"\n[seed] Final counts: {counts}")
        print("[seed] Done! The API will now return real data.")

    finally:
        db.close()


if __name__ == "__main__":
    main()