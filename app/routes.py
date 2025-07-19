"""
API Routes για το FAQ Service.

Αυτό το αρχείο περιέχει όλα τα endpoints.
Το κρατάμε ξεχωριστό από το main.py για καλύτερη οργάνωση.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import logging
import os

from .database import get_db
from .models import Question
from .schemas import QuestionRequest, AnswerResponse, QuestionHistory
from .llm_service import llm_service
from .rag_service import HybridRAGService

# Ρυθμίζουμε το logging
logger = logging.getLogger(__name__)

# Δημιουργούμε ένα router αντί για app
# Αυτό θα το "κολλήσουμε" στο main app αργότερα
router = APIRouter(
    prefix="/api/v1",  # Όλα τα endpoints θα ξεκινούν με /api/v1
    tags=["faq"],      # Για την οργάνωση στο documentation
)

# Initialize RAG service - θα γίνει μία φορά κατά την εκκίνηση
rag_service = None

def get_rag_service():
    """Lazy initialization του RAG service."""
    global rag_service
    if rag_service is None:
        knowledge_base_path = os.path.join("data", "knowledge_base.txt")
        if not os.path.exists(knowledge_base_path):
            raise FileNotFoundError(f"Knowledge base not found at {knowledge_base_path}")
        rag_service = HybridRAGService(knowledge_base_path)
    return rag_service


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest, 
    db: Session = Depends(get_db)
):
    """
    Ask a question to the FAQ system.
    
    Η διαδικασία:
    1. Λαμβάνουμε την ερώτηση από τον χρήστη
    2. Χρησιμοποιούμε το RAG για να βρούμε σχετικό context
    3. Στέλνουμε την ερώτηση και το context στο LLM
    4. Αποθηκεύουμε την ερώτηση και απάντηση στη βάση
    5. Επιστρέφουμε την απάντηση
    
    Args:
        request: Η ερώτηση του χρήστη
        db: Database session (injected automatically)
    
    Returns:
        AnswerResponse με την απάντηση του AI
    """
    try:
        # Παίρνουμε το RAG service
        rag = get_rag_service()
        
        logger.info(f"📝 Processing question: {request.question[:50]}...")
        
        # Χρησιμοποιούμε το RAG για να βρούμε relevant context
        context = rag.get_context_for_llm(request.question)
        
        logger.info(f"🔍 RAG found relevant context ({len(context)} chars)")
        
        # Παίρνουμε απάντηση από το LLM με το context
        answer_text = llm_service.generate_answer_with_context(
            question=request.question,
            context=context
        )
        
        # Δημιουργούμε νέο record στη βάση
        db_question = Question(
            question_text=request.question,
            answer_text=answer_text
        )
        
        # Αποθηκεύουμε στη βάση
        db.add(db_question)
        db.commit()
        db.refresh(db_question)  # Για να πάρουμε το generated ID
        
        logger.info(f"💾 Saved Q&A to database (ID: {db_question.id})")
        
        # Επιστρέφουμε την απάντηση
        return AnswerResponse(answer=answer_text)
        
    except Exception as e:
        # Αν κάτι πάει στραβά, κάνουμε rollback
        db.rollback()
        # Και επιστρέφουμε user-friendly error
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your question: {str(e)}"
        )


@router.get("/history", response_model=List[QuestionHistory])
async def get_history(
    n: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Number of questions to return"
    ),
    db: Session = Depends(get_db)
):
    """
    Get the history of recent questions and answers.
    
    Επιστρέφει τις πιο πρόσφατες ερωτήσεις, με τις νεότερες πρώτες.
    Μπορείς να ελέγξεις πόσες θες με το parameter 'n'.
    
    Args:
        n: Αριθμός ερωτήσεων που θέλεις (1-100, default: 10)
        db: Database session
    
    Returns:
        List of QuestionHistory objects
    """
    try:
        # Query στη βάση - παίρνουμε τις τελευταίες n ερωτήσεις
        questions = db.query(Question)\
            .order_by(Question.timestamp.desc())\
            .limit(n)\
            .all()
        
        # Το FastAPI θα μετατρέψει αυτόματα τα SQLAlchemy objects
        # σε QuestionHistory schemas χάρη στο from_attributes=True
        return questions
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving history: {str(e)}"
        )


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """
    Get statistics about the FAQ system.
    
    Αυτό είναι ένα bonus endpoint που δείχνει χρήσιμα στατιστικά.
    Καλή πρακτική για monitoring και debugging.
    """
    try:
        total_questions = db.query(Question).count()
        
        # Βρίσκουμε την πιο πρόσφατη ερώτηση
        latest_question = db.query(Question)\
            .order_by(Question.timestamp.desc())\
            .first()
        
        return {
            "total_questions": total_questions,
            "latest_question_time": latest_question.timestamp if latest_question else None,
            "api_version": "1.0.0",
            "status": "operational"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving stats: {str(e)}"
        )


@router.get("/llm/test")
async def test_llm_connection():
    """
    Test the connection to the LLM service.
    
    Αυτό είναι χρήσιμο για debugging και για να βεβαιωθείς
    ότι το Ollama τρέχει και το Mistral είναι διαθέσιμο.
    """
    try:
        test_result = llm_service.test_connection()
        return test_result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM test failed: {str(e)}"
        )


@router.post("/rag/search")
async def search_knowledge_base(request: QuestionRequest):
    """
    Εκτελεί RAG search χωρίς να καλέσει το LLM.
    
    Αυτό το endpoint είναι χρήσιμο για:
    - Debugging του RAG system
    - Κατανόηση του πώς λειτουργεί το hybrid search
    - Testing χωρίς να περιμένεις το LLM
    
    Returns:
        Detailed search results με scores και explanations
    """
    try:
        rag = get_rag_service()
        
        # Εκτέλεση hybrid search
        results = rag.search(request.question, n_results=5)
        
        # Μετατροπή αποτελεσμάτων σε JSON-friendly format
        search_results = []
        for result in results:
            search_results.append({
                "question": result.qa_pair.question,
                "answer": result.qa_pair.answer[:200] + "..." if len(result.qa_pair.answer) > 200 else result.qa_pair.answer,
                "scores": {
                    "semantic": round(result.semantic_score, 3),
                    "keyword": round(result.keyword_score, 3),
                    "combined": round(result.combined_score, 3)
                },
                "match_type": result.match_type,
                "explanation": result.explanation
            })
        
        # Παίρνουμε και explanation για το search
        explanation = rag.explain_results(request.question)
        
        return {
            "query": request.question,
            "results": search_results,
            "explanation": explanation,
            "context_preview": rag.get_context_for_llm(request.question)[:500] + "..."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG search failed: {str(e)}"
        )
async def get_stats(db: Session = Depends(get_db)):
    """
    Get statistics about the FAQ system.
    
    Αυτό είναι ένα bonus endpoint που δείχνει χρήσιμα στατιστικά.
    Καλή πρακτική για monitoring και debugging.
    """
    try:
        total_questions = db.query(Question).count()
        
        # Βρίσκουμε την πιο πρόσφατη ερώτηση
        latest_question = db.query(Question)\
            .order_by(Question.timestamp.desc())\
            .first()
        
        return {
            "total_questions": total_questions,
            "latest_question_time": latest_question.timestamp if latest_question else None,
            "api_version": "1.0.0",
            "status": "operational"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving stats: {str(e)}"
        )