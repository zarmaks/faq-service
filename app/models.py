"""
Database models για το FAQ service.
Αυτά είναι τα "καλούπια" που ορίζουν πώς θα αποθηκεύονται τα δεδομένα.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from app.database import Base

class Question(Base):
    """
    Model για την αποθήκευση ερωτήσεων και απαντήσεων.
    Κάθε φορά που κάποιος ρωτάει κάτι, δημιουργείται ένα νέο Question object.
    """
    __tablename__ = "questions"
    
    # Primary key - μοναδικό ID για κάθε ερώτηση
    id = Column(Integer, primary_key=True, index=True)
    
    # Η ερώτηση του χρήστη
    question_text = Column(Text, nullable=False)
    
    # Η απάντηση από το LLM
    answer_text = Column(Text, nullable=False)
    
    # Πότε έγινε η ερώτηση (αυτόματα συμπληρώνεται)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    source = Column(String(50), nullable=True, default="api")
    
    def __repr__(self):
        """Για debugging - δείχνει όμορφα το object"""
        return f"<Question(id={self.id}, question='{self.question_text[:30]}...')>"