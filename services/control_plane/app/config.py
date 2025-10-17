"""
Configuration for the control plane application.
Database setup and environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Fallback to SQLite if DATABASE_URL not set in .env
    DB_DIR = Path(__file__).parent.parent.parent / "data"
    DB_DIR.mkdir(exist_ok=True)
    DATABASE_URL = f"sqlite:///{DB_DIR}/control_plane.db"

# SQLAlchemy engine options
SQLALCHEMY_KWARGS = {
    "connect_args": {"check_same_thread": False}  # Required for SQLite
}

print("Database URL loaded from .env:", DATABASE_URL)