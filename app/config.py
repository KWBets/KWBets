import json
import os
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    app_name: str = "DoubleDown AI"
    environment: str = "development"
    log_level: str = "info"

    # Database
    # NOTE: Plain string default — pydantic-settings resolves DATABASE_URL
    # from the runtime env (highest priority) or .env file automatically.
    # Do NOT use os.getenv() here — it pre-resolves at class definition time
    # and defeats pydantic's env-var-overrides-default behavior.
    database_url: str = "sqlite:///data/doubledown.db"

    # The Odds API
    # NOTE: Plain default — pydantic-settings resolves ODDS_API_KEY from env.
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    # Schedule intervals (in hours)
    odds_fetch_interval_hours: int = 6  # Changed from 1h to 6h to conserve API quota (190 req remaining)
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

    # US-only sportsbook allowlist — regulated/licensed books only
    # Env override via US_BOOKMAKERS env var: accepts both JSON arrays
    # (e.g. ["draftkings","fanduel"]) and comma-separated strings
    # (e.g. "draftkings,fanduel").
    # NOTE: str type so pydantic-settings doesn't JSON-decode the env var;
    # the validator converts to list[str] after pydantic resolves the value.
    us_bookmakers: str = (
        "draftkings,fanduel,betmgm,betrivers,"
        "ballybet,hardrockbet,betparx,fliff"
    )

    @field_validator("us_bookmakers", mode="after")
    @classmethod
    def parse_us_bookmakers(cls, v: str) -> list[str]:
        """Accept both JSON arrays and comma-separated strings for US_BOOKMAKERS."""
        v = v.strip()
        if v.startswith("["):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return [s.strip() for s in v.split(",") if s.strip()]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# Post-process: Railway Postgres sets DATABASE_URL as postgres:// but
# SQLAlchemy requires postgresql://. Fix it here so all consumers of
# settings.database_url see the corrected URL.
if settings.database_url and settings.database_url.startswith("postgres://"):
    settings.database_url = settings.database_url.replace("postgres://", "postgresql://", 1)