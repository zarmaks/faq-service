"""
ChromaDB Service - Vector database για semantic search.

Το ChromaDB αποθηκεύει embeddings και κάνει γρήγορες αναζητήσεις
βάσει σημασιολογικής ομοιότητας. Είναι το "μυαλό" του RAG μας
που θυμάται και βρίσκει σχετικές πληροφορίες.

Αυτή η έκδοση χρησιμοποιεί COSINE distance για καλύτερα semantic search results.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Αποτέλεσμα αναζήτησης από το ChromaDB.
    
    Περιέχει όλες τις πληροφορίες που χρειαζόμαστε
    για να χρησιμοποιήσουμε το αποτέλεσμα.
    """
    qa_id: int          # ID του Q&A pair
    text: str           # Το πλήρες κείμενο
    question: str       # Η ερώτηση
    answer: str         # Η απάντηση  
    similarity: float   # Πόσο κοντά είναι στην αναζήτηση (0-1)
    

class ChromaDBService:
    """
    Service για διαχείριση της vector database.
    
    Αυτή η κλάση διαχειρίζεται:
    - Αποθήκευση embeddings με metadata
    - Semantic search με similarity scoring
    - Διαχείριση της collection
    """
    
    def __init__(self, collection_name: str = "faq_embeddings"):
        """
        Initialize ChromaDB service.
        
        Args:
            collection_name: Όνομα της collection (σαν table σε SQL)
        """
        # Δημιουργούμε client με persistent storage
        # Τα δεδομένα θα αποθηκεύονται στο ./chroma_db
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,  # Απενεργοποίηση telemetry
                allow_reset=True,            # Επιτρέπει reset της collection
                is_persistent=True           # Persistent storage
            )
        )
        
        self.collection_name = collection_name
        self._init_collection()
        
    def _init_collection(self):
        """
        Αρχικοποιεί ή ανακτά την collection.
        
        Η collection είναι σαν ένας πίνακας που αποθηκεύει
        embeddings μαζί με τα metadata τους.
        
        ΣΗΜΑΝΤΙΚΟ: Χρησιμοποιούμε cosine distance αντί για L2
        για καλύτερα semantic search αποτελέσματα.
        """
        try:
            # Προσπαθούμε να πάρουμε υπάρχουσα collection
            self.collection = self.client.get_collection(self.collection_name)
            
            # Ελέγχουμε αν χρησιμοποιεί cosine distance
            # Αν όχι, θα πρέπει να τη διαγράψουμε και να τη ξαναφτιάξουμε
            metadata = self.collection.metadata
            if metadata.get("hnsw:space") != "cosine":
                logger.warning("⚠️  Existing collection uses L2 distance, not cosine!")
                logger.info("🔄 Recreating collection with cosine distance...")
                self.client.delete_collection(self.collection_name)
                raise Exception("Need to recreate with cosine")
            
            logger.info(f"✅ Loaded existing collection '{self.collection_name}' (cosine distance)")
            logger.info(f"   Contains {self.collection.count()} embeddings")
            
        except:
            # Δημιουργούμε νέα collection με cosine distance
            # Το cosine distance είναι καλύτερο για semantic similarity
            # γιατί μετράει τη γωνία μεταξύ των vectors, όχι την απόσταση
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "FAQ Q&A pairs embeddings with cosine similarity",
                    "hnsw:space": "cosine"  # Καθορίζουμε cosine distance metric
                }
            )
            logger.info(f"✅ Created new collection '{self.collection_name}' with COSINE distance")
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        qa_pairs: List[Dict],
        force_reset: bool = False
    ):
        """
        Προσθέτει embeddings στη collection.
        
        Κάθε embedding αποθηκεύεται μαζί με metadata που μας
        επιτρέπουν να ανακτήσουμε το αρχικό Q&A pair.
        
        Args:
            embeddings: List με τα embedding vectors
            qa_pairs: List με τα Q&A pair objects (as dicts)
            force_reset: Αν True, διαγράφει τα παλιά δεδομένα
        """
        if force_reset:
            # Διαγράφουμε και ξαναδημιουργούμε τη collection
            self.client.delete_collection(self.collection_name)
            self._init_collection()
            logger.info("🔄 Reset collection")
        
        # Προετοιμάζουμε τα δεδομένα για το ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for i, qa in enumerate(qa_pairs):
            # Μοναδικό ID για κάθε embedding
            ids.append(f"qa_{qa['id']}")
            
            # Το πλήρες κείμενο (αυτό θα εμφανίζεται στα αποτελέσματα)
            documents.append(qa['full_text'])
            
            # Metadata για να μπορούμε να ανακτήσουμε πληροφορίες
            metadatas.append({
                "qa_id": qa['id'],
                "question": qa['question'],
                "answer": qa['answer']
            })
        
        # Προσθήκη στο ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✅ Added {len(embeddings)} embeddings to ChromaDB")
        logger.info(f"   Total embeddings in collection: {self.collection.count()}")
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 3
    ) -> List[SearchResult]:
        """
        Κάνει semantic search στη collection.
        
        Βρίσκει τα embeddings που είναι πιο "κοντά" στο query
        embedding, δηλαδή τα πιο σημασιολογικά παρόμοια.
        
        ΣΗΜΕΙΩΣΗ: Με cosine distance, μικρότερη τιμή = μεγαλύτερη ομοιότητα
        - Distance 0 = πανομοιότυπα vectors (cosine similarity = 1)
        - Distance 1 = ορθογώνια vectors (cosine similarity = 0)
        - Distance 2 = αντίθετα vectors (cosine similarity = -1)
        
        Args:
            query_embedding: Το embedding της ερώτησης
            n_results: Πόσα αποτελέσματα να επιστρέψει
            
        Returns:
            List of SearchResult objects, ταξινομημένα κατά similarity
        """
        # Query στο ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Μετατροπή αποτελεσμάτων σε SearchResult objects
        search_results = []
        
        # Το ChromaDB επιστρέφει lists of lists (για batch queries)
        # Εμείς έχουμε μόνο ένα query, οπότε παίρνουμε το πρώτο
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        # Ας καταλάβουμε τι distances παίρνουμε
        logger.debug(f"🔍 Raw cosine distances: {distances}")
        
        for i in range(len(documents)):
            # Μετατροπή cosine distance σε similarity score
            # Το cosine distance στο ChromaDB κυμαίνεται από 0 έως 2
            # όπου 0 = ίδια vectors, 1 = ορθογώνια, 2 = αντίθετα
            distance = distances[i]
            
            # Μετατροπή σε similarity (0-1 range)
            # Για normalized embeddings, το cosine similarity = 1 - (cosine_distance / 2)
            # Αλλά το ChromaDB μπορεί να χρησιμοποιεί διαφορετικό scaling
            
            # Ας ελέγξουμε αν οι αποστάσεις είναι στο αναμενόμενο range [0, 2]
            if distance > 2:
                # Αν οι αποστάσεις είναι μεγαλύτερες από 2, το ChromaDB
                # μπορεί να χρησιμοποιεί διαφορετικό scaling
                logger.warning(f"⚠️  Unexpected cosine distance > 2: {distance}")
                # Χρησιμοποιούμε εναλλακτική μετατροπή
                similarity = 1.0 / (1.0 + distance)
            else:
                # Κανονική μετατροπή για cosine distance [0, 2]
                # distance = 0 → similarity = 1
                # distance = 1 → similarity = 0.5
                # distance = 2 → similarity = 0
                similarity = 1.0 - (distance / 2.0)
            
            # Βεβαιωνόμαστε ότι το similarity είναι στο [0, 1]
            similarity = max(0.0, min(1.0, similarity))
            
            # Log για debugging
            logger.debug(f"   Result {i+1}: distance={distance:.4f}, "
                        f"similarity={similarity:.3f}, "
                        f"question='{metadatas[i]['question'][:50]}...'")
            
            result = SearchResult(
                qa_id=metadatas[i]['qa_id'],
                text=documents[i],
                question=metadatas[i]['question'],
                answer=metadatas[i]['answer'],
                similarity=float(similarity)
            )
            search_results.append(result)
        
        return search_results
    
    def get_stats(self) -> Dict:
        """
        Επιστρέφει στατιστικά για τη collection.
        
        Χρήσιμο για debugging και monitoring.
        """
        count = self.collection.count()
        
        # Παίρνουμε ένα sample για να δούμε τη δομή
        sample = None
        if count > 0:
            sample_results = self.collection.peek(1)
            if sample_results['documents']:
                sample = {
                    "document_preview": sample_results['documents'][0][:100] + "...",
                    "metadata": sample_results['metadatas'][0]
                }
        
        # Προσπαθούμε να πάρουμε πληροφορίες για το distance metric
        try:
            metadata = self.collection.metadata
            distance_metric = metadata.get("hnsw:space", "unknown")
        except:
            distance_metric = "unknown"
        
        return {
            "collection_name": self.collection_name,
            "total_embeddings": count,
            "distance_metric": distance_metric,
            "sample": sample,
            "storage_path": "./chroma_db"
        }
    
    def clear_collection(self):
        """
        Καθαρίζει όλα τα δεδομένα από τη collection.
        
        Χρήσιμο για testing ή reset.
        """
        self.client.delete_collection(self.collection_name)
        self._init_collection()
        logger.info("🧹 Cleared collection")
    
    def find_similar_questions(
        self,
        question: str,
        query_embedding: List[float],
        threshold: float = 0.7
    ) -> List[str]:
        """
        Βρίσκει παρόμοιες ερωτήσεις που έχουν ήδη απαντηθεί.
        
        Χρήσιμο για να προτείνουμε: "Μήπως εννοείτε..."
        
        Args:
            question: Η ερώτηση του χρήστη
            query_embedding: Το embedding της ερώτησης
            threshold: Minimum similarity για να θεωρηθεί σχετική
            
        Returns:
            List με παρόμοιες ερωτήσεις
        """
        results = self.search(query_embedding, n_results=5)
        
        similar_questions = []
        for result in results:
            if result.similarity >= threshold and result.question.lower() != question.lower():
                similar_questions.append(result.question)
        
        return similar_questions[:3]  # Max 3 προτάσεις