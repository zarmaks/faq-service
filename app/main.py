"""
Main FastAPI application entry point.

Αυτό το αρχείο είναι υπεύθυνο μόνο για:
1. Την αρχικοποίηση της εφαρμογής
2. Τη σύνδεση των routes
3. Τη δημιουργία της βάσης δεδομένων
4. Τα γενικά settings

Η πραγματική λογική βρίσκεται στα άλλα modules.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .database import engine
from .models import Base
from .routes import router

# Ρύθμιση logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Δημιουργούμε τους πίνακες στη βάση (αν δεν υπάρχουν ήδη)
Base.metadata.create_all(bind=engine)

# Δημιουργούμε την κύρια FastAPI εφαρμογή
app = FastAPI(
    title="FAQ Service API",
    description="AI-powered FAQ service with query logging and retrieval",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc - εναλλακτικό documentation
)

# Προσθέτουμε CORS middleware (επιτρέπει requests από browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Στο production, βάλε συγκεκριμένα origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Συνδέουμε τα routes από το routes.py
app.include_router(router)


@app.get("/")
async def root():
    """
    Root endpoint - Καλωσόρισμα και οδηγίες.
    
    Αυτό είναι το μόνο endpoint που μένει στο main.py
    γιατί είναι γενικό και όχι μέρος της business logic.
    """
    return {
        "message": "Welcome to FAQ Service API",
        "documentation": "/docs",
        "api_endpoints": {
            "POST /api/v1/ask": "Ask a question",
            "GET /api/v1/history": "Get question history",
            "GET /api/v1/stats": "Get system statistics"
        },
        "health_check": "/health"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Χρήσιμο για:
    - Docker health checks
    - Load balancer checks
    - Monitoring systems
    """
    return {
        "status": "healthy",
        "service": "FAQ Service",
        "version": "1.0.0"
    }


@app.on_event("startup")
async def startup_event():
    """
    Τρέχει όταν ξεκινάει η εφαρμογή.
    
    Μπορούμε να προσθέσουμε εδώ:
    - Έλεγχο σύνδεσης με τη βάση
    - Προφόρτωση του LLM model
    - Άλλες αρχικοποιήσεις
    """
    print("🚀 FAQ Service is starting up...")
    print("📚 Database tables created/verified")
    print("✅ Ready to serve requests!")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Τρέχει όταν σταματάει η εφαρμογή.
    
    Καλό μέρος για cleanup tasks.
    """
    print("👋 FAQ Service is shutting down...")