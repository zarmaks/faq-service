"""
Embeddings Service - Μετατρέπει κείμενο σε vectors.

Αυτό το module χρησιμοποιεί το Ollama για να δημιουργήσει embeddings.
Τα embeddings είναι μαθηματικές αναπαραστάσεις του νοήματος του κειμένου.
"""

import requests
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import time

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"  # Lightweight embedding model


class EmbeddingsService:
    """
    Service για τη δημιουργία text embeddings μέσω Ollama.
    
    Τα embeddings μας επιτρέπουν να μετρήσουμε τη σημασιολογική
    ομοιότητα μεταξύ κειμένων - πόσο "κοντά" είναι στο νόημα.
    """
    
    def __init__(
        self, 
        base_url: str = OLLAMA_BASE_URL,
        model: str = EMBEDDING_MODEL
    ):
        """
        Initialize embeddings service.
        
        Args:
            base_url: Ollama API URL
            model: Το embedding model που θα χρησιμοποιήσουμε
        """
        self.base_url = base_url
        self.model = model
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """
        Ελέγχει αν το embedding model είναι διαθέσιμο.
        Αν όχι, δίνει οδηγίες για να το κατεβάσει ο χρήστης.
        """
        try:
            # Έλεγχος αν το model υπάρχει
            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            # Έλεγχος αν το model υπάρχει (με ή χωρίς :latest tag)
            if not any(self.model in model for model in model_names):
                logger.warning(f"⚠️  Embedding model '{self.model}' not found!")
                logger.info(f"📥 Please run: ollama pull {self.model}")
                logger.info("This is a small model (~274MB) optimized for embeddings")
                
                # Προσπαθούμε να το κατεβάσουμε αυτόματα
                logger.info("🔄 Attempting to pull model automatically...")
                pull_response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model}
                )
                if pull_response.status_code == 200:
                    logger.info("✅ Model pulled successfully!")
                else:
                    raise RuntimeError(
                        f"Please manually run: ollama pull {self.model}"
                    )
            else:
                logger.info(f"✅ Embedding model '{self.model}' is available")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama. Make sure it's running with: ollama serve"
            )
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Δημιουργεί embedding για ένα κείμενο.
        
        Το embedding είναι ένας πίνακας αριθμών που αναπαριστά
        το νόημα του κειμένου. Παρόμοια κείμενα θα έχουν
        παρόμοια embeddings (μικρή απόσταση μεταξύ τους).
        
        Args:
            text: Το κείμενο που θα μετατραπεί
            
        Returns:
            List of floats (το embedding vector)
        """
        try:
            # Καθαρίζουμε το κείμενο
            text = text.strip()
            if not text:
                raise ValueError("Cannot create embedding for empty text")
            
            # API call στο Ollama
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            
            # Παίρνουμε το embedding
            embedding = response.json()["embedding"]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    def create_embeddings_batch(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Δημιουργεί embeddings για πολλά κείμενα.
        
        Επεξεργάζεται τα κείμενα ένα-ένα με progress indication.
        Σε production θα χρησιμοποιούσαμε true batching,
        αλλά το Ollama δεν το υποστηρίζει ακόμα.
        
        Args:
            texts: List με τα κείμενα
            show_progress: Αν θα δείχνει progress bar
            
        Returns:
            List of embeddings (κάθε embedding είναι list of floats)
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            if show_progress:
                print(f"\r📊 Creating embeddings: {i+1}/{len(texts)}", end="")
            
            # Μικρή καθυστέρηση για να μην πνίξουμε το Ollama
            if i > 0:
                time.sleep(0.1)
            
            embedding = self.create_embedding(text)
            embeddings.append(embedding)
        
        if show_progress:
            print(f"\r✅ Created {len(embeddings)} embeddings successfully!")
        
        return embeddings
    
    def cosine_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Υπολογίζει την ομοιότητα μεταξύ δύο embeddings.
        
        Χρησιμοποιούμε cosine similarity που μετράει τη γωνία
        μεταξύ δύο vectors. Το αποτέλεσμα είναι από -1 έως 1:
        - 1: Πανομοιότυπα κείμενα
        - 0: Άσχετα κείμενα  
        - -1: Αντίθετα κείμενα (σπάνιο)
        
        Args:
            embedding1: Πρώτο embedding vector
            embedding2: Δεύτερο embedding vector
            
        Returns:
            Similarity score (0 έως 1 συνήθως)
        """
        # Μετατροπή σε numpy arrays για ευκολότερους υπολογισμούς
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity = dot product / (norm1 * norm2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Αποφυγή διαίρεσης με μηδέν
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    def find_most_similar(
        self, 
        query_embedding: List[float],
        embeddings_db: List[List[float]],
        top_k: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Βρίσκει τα πιο παρόμοια embeddings από μια βάση.
        
        Args:
            query_embedding: Το embedding της ερώτησης
            embeddings_db: List με όλα τα embeddings της βάσης
            top_k: Πόσα αποτελέσματα να επιστρέψει
            
        Returns:
            List of tuples (index, similarity_score)
        """
        similarities = []
        
        # Υπολογίζουμε similarity με κάθε embedding στη βάση
        for i, db_embedding in enumerate(embeddings_db):
            similarity = self.cosine_similarity(query_embedding, db_embedding)
            similarities.append((i, similarity))
        
        # Ταξινομούμε κατά φθίνουσα σειρά similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Επιστρέφουμε τα top_k
        return similarities[:top_k]
    
    def test_similarity(self):
        """
        Test function για να δούμε πώς λειτουργούν τα embeddings.
        
        Δοκιμάζει παρόμοιες και διαφορετικές προτάσεις.
        """
        print("🧪 Testing Embeddings Similarity\n")
        
        test_pairs = [
            ("What is the refund policy?", "How can I get a refund?"),
            ("What is the refund policy?", "What is your return policy?"),
            ("What is the refund policy?", "How do I deploy my app?"),
            ("password reset", "forgot password"),
            ("API rate limits", "How many API calls can I make?"),
        ]
        
        for text1, text2 in test_pairs:
            emb1 = self.create_embedding(text1)
            emb2 = self.create_embedding(text2)
            similarity = self.cosine_similarity(emb1, emb2)
            
            print(f"📝 '{text1}' vs '{text2}'")
            print(f"   Similarity: {similarity:.3f}")
            print(f"   Interpretation: {'Very similar' if similarity > 0.8 else 'Somewhat similar' if similarity > 0.5 else 'Different'}")
            print()


# Utility function
def test_embeddings():
    """Γρήγορο test για τα embeddings."""
    service = EmbeddingsService()
    service.test_similarity()