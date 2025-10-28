import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Header
from .db.models import CustomerCreateRequest, APIKeyCreateRequest, CreditPurchaseRequest