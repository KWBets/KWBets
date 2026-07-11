import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    app_name: str = "DoubleDown AI"
    environment: str = "development"
    log_level: str = "info"

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL", "sqlite:///data/doubledown.db"
    )

    # The Odds API
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    # Schedule intervals (in hours)
    odds_fetch_interval_hours: int = 1
    model_retrain_interval_hours: int = 24

    # Sports supported (The Odds API keys)
    supported_sports: list[str] = [
        "americanfootball_nfl",
        "americanfootball_ncaaf",
        "baseball_mlb",
        "basketball_nba",
        "basketball_ncaab",
        "icehockey_nhl",
        "soccer_fifa_world_cup",
        "soccer_usa_mls",
        "soccer_epl",
        "soccer_uefa_champs_league",
        "soccer_la_liga",
        "soccer_bundesliga",
        "golf_pga_championship",
        "golf_us_open",
        "tennis_atp_wimbledon",
        "tennis_wta_wimbledon",
    ]

    # Market types
    supported_markets: list[str] = ["h2h", "spreads", "totals"]

    # Feature flags
    feature_flags: dict = {
        "show_model_ev": False,  # Hide model edge/EV by default
    }

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()