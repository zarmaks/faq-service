"""
Database configuration and session management.
Αυτό το αρχείο ρυθμίζει τη σύνδεση με τη βάση δεδομένων.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Δημιουργούμε τη σύνδεση με SQLite
# Το sqlite:///./faq.db σημαίνει: χρησιμοποίησε SQLite και αποθήκευσε στο αρχείο faq.db
SQLALCHEMY_DATABASE_URL = "sqlite:///./faq.db"

# Δημιουργούμε τη "μηχανή" που θα διαχειρίζεται τη σύνδεση
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Απαραίτητο για SQLite
)

# Δημιουργούμε το SessionLocal - αυτό θα χρησιμοποιούμε για κάθε database operation
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class για τα models μας - όλα τα tables θα κληρονομούν από αυτό
Base = declarative_base()

# Dependency για το FastAPI - δίνει μας database session για κάθε request
def get_db():
    """
    Δημιουργεί μια database session για κάθε request.
    Κλείνει αυτόματα όταν τελειώσει το request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()