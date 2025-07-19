"""
Pydantic schemas για validation και serialization.

Αυτά τα models ορίζουν το "συμβόλαιο" μεταξύ client και server:
- Τι δεδομένα περιμένουμε να λάβουμε (request schemas)
- Τι δεδομένα θα επιστρέψουμε (response schemas)
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime


class QuestionRequest(BaseModel):
    """
    Schema για το request όταν κάποιος κάνει μια ερώτηση.
    
    Χρησιμοποιούμε το Field για να προσθέσουμε validation και documentation.
    """
    question: str = Field(
        ...,  # ... σημαίνει required field
        min_length=5,
        max_length=500,
        description="The question to ask the FAQ system",
        #example="What is your refund policy?"
    )

    source: Optional[str] = Field(
        default="api",
        pattern="^(web|api|mobile)$",  # Μόνο αυτές οι τιμές επιτρέπονται
        description="The source of the question (e.g., 'api', 'web', etc.)"
    )

    # Μπορούμε να προσθέσουμε custom validation
    @validator('question')
    def validate_question(cls, v):
        """Καθαρίζει την ερώτηση από περιττά κενά"""
        return v.strip()


class AnswerResponse(BaseModel):
    """
    Schema για την απάντηση που επιστρέφουμε.
    
    Κρατάμε το απλό - μόνο η απάντηση.
    """
    answer: str = Field(
        ...,
        description="The AI-generated answer",
        #example="Our refund policy allows returns within 30 days..."
    )


class QuestionHistory(BaseModel):
    """
    Schema για το ιστορικό ερωτήσεων.
    
    Περιλαμβάνει όλες τις πληροφορίες που έχουμε αποθηκεύσει.
    """
    id: int = Field(..., description="Unique identifier")
    question_text: str = Field(..., description="The original question")
    answer_text: str = Field(..., description="The generated answer")
    timestamp: datetime = Field(..., description="When the question was asked")
    source: Optional[str] = Field(
        default="api",
        description="The source of the question (e.g., 'api', 'web', etc.)"
    )
    
    class Config:
        """
        Η ρύθμιση from_attributes επιτρέπει στο Pydantic 
        να διαβάζει δεδομένα από SQLAlchemy objects.
        """
        from_attributes = True


class HistoryQueryParams(BaseModel):
    """
    Schema για τα query parameters του /history endpoint.
    
    Αυτό μας επιτρέπει να έχουμε καλύτερο validation και documentation.
    """
    n: int = Field(
        default=10,
        ge=1,  # greater or equal to 1
        le=100,  # less or equal to 100
        description="Number of recent questions to return"
    )