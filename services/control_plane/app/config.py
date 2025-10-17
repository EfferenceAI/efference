"""
This is just configuration for the database and other services we might use
"""
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

print("Database URL loaded from .env:", DATABASE_URL)