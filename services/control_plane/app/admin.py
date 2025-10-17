import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Header
from .models import CustomerCreateRequest, APIKeyCreateRequest, CreditPurchaseRequest