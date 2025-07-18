"""
LLM Service για επικοινωνία με Ollama/Mistral.

Αυτό το module είναι υπεύθυνο για:
1. Την επικοινωνία με το Ollama API
2. Το prompt engineering
3. Τη διαχείριση του context
"""

import json
import requests
from typing import Dict, Any
import logging

# Ρυθμίζουμε το logging για debugging
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"
DEFAULT_TEMPERATURE = 0.7  # Πόσο "δημιουργικό" θα είναι το model (0-1)
DEFAULT_MAX_TOKENS = 500   # Μέγιστο μήκος απάντησης


class LLMService:
    """
    Service class για την επικοινωνία με το Ollama API.
    
    Αυτή η κλάση κάνει abstract την επικοινωνία με το LLM,
    ώστε ο υπόλοιπος κώδικας να μην χρειάζεται να ξέρει
    τις λεπτομέρειες του πώς λειτουργεί το Ollama.
    """
    
    def __init__(
        self, 
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """
        Initialize the LLM service.
        
        Args:
            base_url: Το URL όπου τρέχει το Ollama
            model: Το model που θα χρησιμοποιήσουμε
            temperature: Πόσο "τυχαίες" θα είναι οι απαντήσεις
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self._check_connection()

    def _check_connection(self) -> None:
        """Check if we can connect to Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            logger.info(f"✅ Connected to Ollama at {self.base_url}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            logger.error("Make sure Ollama is running with: ollama serve")
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}"
            )

    def create_system_prompt(self, knowledge_base: str) -> str:
        """Create the system prompt for the AI."""
        return (
            "You are a helpful customer support assistant for CloudSphere Platform.\n\n"
            "Your role is to answer customer questions based ONLY on the following knowledge base.\n"
            "If a question cannot be answered using the knowledge base, politely say that you don't have that information.\n\n"
            "IMPORTANT RULES:\n"
            "1. Only use information from the knowledge base below\n"
            "2. Be concise and direct in your answers\n"
            "3. If you're not sure, say so - don't make up information\n"
            "4. Be friendly and professional\n"
            "5. Format your answers clearly with proper punctuation\n"
            "6. If the question is about something not in the knowledge base, respond with: \"I don't have information about that in my knowledge base.\"\n"
            "   Please contact support@cloudsphere.com for assistance with topics not covered here.\n"
            "7. If someone asks a general greeting (hello, hi, etc.), respond professionally but remind them you're here to answer CloudSphere-related questions\n\n"
            "KNOWLEDGE BASE:\n"
            f"{knowledge_base}\n\n"
            "Remember: You are ONLY a CloudSphere support assistant. Do not answer questions unrelated to CloudSphere or provide information not in the knowledge base above."
        )
    
    def generate_answer(
        self,
        question: str,
        knowledge_base: str,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
        """
        Στέλνει ερώτηση στο LLM και παίρνει απάντηση.
        
        Η διαδικασία:
        1. Δημιουργούμε το system prompt με το knowledge base
        2. Συνδυάζουμε με την ερώτηση του χρήστη
        3. Στέλνουμε στο Ollama
        4. Επιστρέφουμε την απάντηση
        
        Args:
            question: Η ερώτηση του χρήστη
            knowledge_base: Το περιεχόμενο που θα χρησιμοποιήσει για context
            max_tokens: Μέγιστο μήκος απάντησης
            
        Returns:
            Η απάντηση από το LLM
        """
        try:
            # Δημιουργούμε το πλήρες prompt
            system_prompt = self.create_system_prompt(knowledge_base)
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Customer Question: {question}\n\n"
                "Assistant Answer:"
            )
            
            # Προετοιμάζουμε το request για το Ollama
            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": full_prompt,
                "temperature": self.temperature,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "stop": [
                        "Customer Question:",
                        "\n\n"
                    ]
                }
            }
            
            # Log για debugging
            logger.info(f"📤 Sending question to LLM: {question[:50]}...")
            
            # Στέλνουμε το request
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30  # 30 seconds timeout
            )
            response.raise_for_status()
            
            # Παίρνουμε την απάντηση
            result = response.json()
            answer = result.get("response", "").strip()
            
            # Αν η απάντηση είναι κενή, κάτι πήγε στραβά
            if not answer:
                logger.error("Empty response from LLM")
                return (
                    "I apologize, but I couldn't generate an answer. "
                    "Please try again."
                )
            
            logger.info(f"✅ Received answer from LLM ({len(answer)} chars)")
            return answer
            
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return (
                "The request took too long. "
                "Please try again with a simpler question."
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return (
                "I'm having trouble connecting to the AI service. "
                "Please try again later."
            )
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return "An unexpected error occurred. Please try again."
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Τεστάρει τη σύνδεση και επιστρέφει πληροφορίες.
        
        Χρήσιμο για debugging και health checks.
        """
        try:
            # Έλεγχος models
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json()
            
            # Απλό test generation
            test_response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "Say 'Hello, I am working!'",
                    "stream": False
                }
            )
            test_response.raise_for_status()
            
            return {
                "status": "connected",
                "ollama_url": self.base_url,
                "model": self.model,
                "available_models": [
                    m["name"] for m in models.get("models", [])
                ],
                "test_response": test_response.json().get(
                    "response",
                    "No response"
                )
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e),
                "ollama_url": self.base_url
            }


# Δημιουργούμε ένα global instance του service
# Αυτό θα χρησιμοποιηθεί από τα routes
llm_service = LLMService()