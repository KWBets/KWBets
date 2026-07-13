from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import settings
import os
# numpy 2.x compatibility: psycopg2 stringifies np.float64 as "np.float64(...)"
# which breaks SQL inserts. Register adapters to convert numpy scalars to
# plain Python numbers before they reach the database.
import numpy as np
from psycopg2.extensions import register_adapter, AsIs

register_adapter(np.float64, lambda v: AsIs(repr(float(v))))
register_adapter(np.int64, lambda v: AsIs(int(v)))

# Ensure data directory exists (SQLite only — Postgres doesn't need it)
if "sqlite" in settings.database_url:
    os.makedirs("data", exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False,
)

# Enable WAL mode for SQLite for better concurrent access
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if "sqlite" in settings.database_url:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """Dependency for FastAPI to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables."""
    from app.models import (  # noqa: F401 - ensure models are imported
        RawOdds,
        ProcessedFeatures,
        ModelPrediction,
        ValueBet,
        UserAlert,
        ModelRegistry,
        PickOutcome,
        User,
        ReferralEvent,
        ProCreditUsage,
        CreatorEarning,
    )
    Base.metadata.create_all(bind=engine)
