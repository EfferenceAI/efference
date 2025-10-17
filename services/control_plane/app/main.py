"""
Initial setup for data models
"""

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import SQLModel, create_engine, Session
from typing import Optional
#from .models import (CustomerCreateRequest, APIKeyCreateRequest, CreditPurchaseRequest)
from .security import validate_customer_api
from .config import DATABASE_URL

app = FastAPI()

def get_db():
    db = Session(create_engine(DATABASE_URL))
    try:
        yield db
    finally:
        db.close()

