import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_PREFIX = "/api/v1"

# Odds API (https://the-odds-api.com)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_SPORTS = os.getenv("ODDS_SPORTS", "americanfootball_nfl,basketball_nba").split(",")
ODDS_MARKETS = os.getenv("ODDS_MARKETS", "h2h,spreads,totals").split(",")
ODDS_REGIONS = os.getenv("ODDS_REGIONS", "us")

# Model training
TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
MIN_EDGE_THRESHOLD = float(os.getenv("MIN_EDGE_THRESHOLD", "0.03"))

# Alerts
ALERT_CHANNELS = os.getenv("ALERT_CHANNELS", "console").split(",")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Retrain
RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))
MIN_SAMPLES_FOR_RETRAIN = int(os.getenv("MIN_SAMPLES_FOR_RETRAIN", "100"))

# Ensure directories exist
for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
