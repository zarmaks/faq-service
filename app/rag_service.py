"""
Hybrid RAG Service - Συνδυάζει semantic και keyword search.

Αυτό είναι το κεντρικό service που ενορχηστρώνει όλα τα components:
- Parser για να διαβάσει το knowledge base
- Embeddings για semantic         # Βήμα 4: Ταξινόμηση και επιστροφή αποτελεσμάτων
        final_results = list(results_map.values())
        
        # Ταξινόμηση κατά combined score
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Κρατάμε μόνο τα top n_results
        final_results = final_results[:n_results]ng
- ChromaDB για semantic search
- TF-IDF για keyword search
- Scoring και ranking για τα καλύτερα αποτελέσματα
"""

from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import os

from .kb_parser import KnowledgeBaseParser, QAPair
from .embeddings_service import EmbeddingsService
from .chromadb_service import ChromaDBService, SearchResult
from .tfidf_service import TFIDFService

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """
    Το τελικό αποτέλεσμα του hybrid search.
    
    Περιέχει πληροφορίες από όλες τις μεθόδους αναζήτησης
    και ένα combined score που δείχνει τη συνολική σχετικότητα.
    """
    qa_pair: QAPair
    semantic_score: float      # Score από ChromaDB (0-1)
    keyword_score: float       # Score από TF-IDF (0-1)
    combined_score: float      # Weighted combination (0-1)
    match_type: str           # "semantic", "keyword", or "both"
    explanation: str          # Γιατί επιλέχθηκε αυτό το αποτέλεσμα


class HybridRAGService:
    """
    Η κύρια κλάση που συνδυάζει όλα τα RAG components.
    
    Αυτή η κλάση:
    1. Φορτώνει και επεξεργάζεται το knowledge base
    2. Δημιουργεί embeddings και indexes
    3. Εκτελεί hybrid search
    4. Επιστρέφει τα καλύτερα αποτελέσματα για το LLM
    """
    
    def __init__(
        self,
        knowledge_base_path: str,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ):
        """
        Initialize Hybrid RAG Service.
        
        Args:
            knowledge_base_path: Path στο knowledge base file
            semantic_weight: Πόσο βάρος δίνουμε στο semantic search (0-1)
            keyword_weight: Πόσο βάρος δίνουμε στο keyword search (0-1)
        """
        self.kb_path = knowledge_base_path
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        # Επιβεβαίωση ότι τα weights αθροίζουν σε 1
        assert abs(semantic_weight + keyword_weight - 1.0) < 0.01, \
            "Weights must sum to 1.0"
        
        # Initialize όλα τα services
        self.parser = KnowledgeBaseParser(knowledge_base_path)
        self.embeddings_service = EmbeddingsService()
        self.chromadb_service = ChromaDBService()
        self.tfidf_service = TFIDFService()
        
        # Φόρτωση και indexing του knowledge base
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """
        Φορτώνει το knowledge base και δημιουργεί όλα τα indexes.
        
        Αυτή η διαδικασία τρέχει μία φορά κατά την εκκίνηση:
        1. Parse Q&A pairs
        2. Create embeddings  
        3. Store in ChromaDB
        4. Create TF-IDF index
        """
        logger.info("🚀 Initializing Hybrid RAG Service...")
        
        # Βήμα 1: Parse knowledge base
        logger.info("📖 Parsing knowledge base...")
        self.qa_pairs = self.parser.parse()
        logger.info(f"   Found {len(self.qa_pairs)} Q&A pairs")
        
        # Βήμα 2: Προετοιμασία texts για indexing
        texts = [qa.to_text() for qa in self.qa_pairs]
        qa_dicts = [qa.to_dict() for qa in self.qa_pairs]
        
        # Βήμα 3: Create embeddings
        logger.info("🧮 Creating embeddings...")
        embeddings = self.embeddings_service.create_embeddings_batch(
            texts, 
            show_progress=True
        )
        
        # Βήμα 4: Store in ChromaDB
        logger.info("💾 Storing in ChromaDB...")
        self.chromadb_service.add_embeddings(
            embeddings, 
            qa_dicts,
            force_reset=True  # Καθαρίζουμε παλιά δεδομένα
        )
        
        # Βήμα 5: Create TF-IDF index
        logger.info("📊 Creating TF-IDF index...")
        self.tfidf_service.fit(texts, qa_dicts)
        
        logger.info("✅ Hybrid RAG Service initialized successfully!")
    
    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> List[HybridSearchResult]:
        """
        Εκτελεί hybrid search για μια ερώτηση.
        
        Η διαδικασία:
        1. Semantic search μέσω ChromaDB
        2. Keyword search μέσω TF-IDF
        3. Συνδυασμός και scoring
        4. Ranking και επιστροφή των καλύτερων
        
        Args:
            query: Η ερώτηση του χρήστη
            n_results: Μέγιστος αριθμός αποτελεσμάτων
            
        Returns:
            List of HybridSearchResult, ταξινομημένα κατά combined score
        """
        logger.info(f"🔍 Hybrid search for: '{query}'")
        
        # Βήμα 1: Semantic Search
        query_embedding = self.embeddings_service.create_embedding(query)
        semantic_results = self.chromadb_service.search(
            query_embedding, 
            n_results=n_results
        )
        
        # Βήμα 2: Keyword Search  
        keyword_results = self.tfidf_service.search(query, n_results=n_results)
        
        # Βήμα 3: Συνδυασμός αποτελεσμάτων
        results_map = {}  # qa_id -> HybridSearchResult
        
        # Προσθήκη semantic results
        for result in semantic_results:
            qa_pair = self.parser.get_by_id(result.qa_id)
            
            hybrid_result = HybridSearchResult(
                qa_pair=qa_pair,
                semantic_score=result.similarity,
                keyword_score=0.0,  # Θα ενημερωθεί αν βρεθεί και στο TF-IDF
                combined_score=result.similarity * self.semantic_weight,
                match_type="semantic",
                explanation=f"Strong semantic match (score: {result.similarity:.2f})"
            )
            
            results_map[result.qa_id] = hybrid_result
        
        # Προσθήκη/ενημέρωση με keyword results
        for qa_id, keyword_score in keyword_results:
            if qa_id in results_map:
                # Υπάρχει ήδη από semantic - ενημερώνουμε
                result = results_map[qa_id]
                result.keyword_score = keyword_score
                result.combined_score = (
                    result.semantic_score * self.semantic_weight +
                    keyword_score * self.keyword_weight
                )
                result.match_type = "both"
                result.explanation = (
                    f"Both semantic ({result.semantic_score:.2f}) "
                    f"and keyword ({keyword_score:.2f}) match"
                )
            else:
                # Νέο αποτέλεσμα μόνο από keyword
                qa_pair = self.parser.get_by_id(qa_id)
                
                hybrid_result = HybridSearchResult(
                    qa_pair=qa_pair,
                    semantic_score=0.0,
                    keyword_score=keyword_score,
                    combined_score=keyword_score * self.keyword_weight,
                    match_type="keyword",
                    explanation=f"Strong keyword match (score: {keyword_score:.2f})"
                )
                
                results_map[qa_id] = hybrid_result
        
        # Βήμα 4: Ταξινόμηση και επιστροφή αποτελεσμάτων
        final_results = list(results_map.values())
        
        # Ταξινόμηση κατά combined score
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Κρατάμε μόνο τα top n_results
        final_results = final_results[:n_results]
        
        # Logging για debugging
        logger.info(f"   Semantic results: {len(semantic_results)}")
        logger.info(f"   Keyword results: {len(keyword_results)}")
        logger.info(f"   Combined results: {len(final_results)}")
        
        for i, result in enumerate(final_results[:3]):
            logger.info(
                f"   #{i+1}: {result.qa_pair.question[:50]}... "
                f"(combined: {result.combined_score:.2f}, type: {result.match_type})"
            )
        
        return final_results
    
    def get_context_for_llm(
        self, 
        query: str, 
        max_context_length: int = 2000
    ) -> str:
        """
        Προετοιμάζει το context για το LLM βάσει της ερώτησης.
        
        Κάνει search και μορφοποιεί τα αποτελέσματα σε κείμενο
        που μπορεί να χρησιμοποιήσει το LLM για να απαντήσει.
        
        Args:
            query: Η ερώτηση του χρήστη
            max_context_length: Μέγιστο μήκος context σε χαρακτήρες
            
        Returns:
            Formatted context string για το LLM
        """
        # Εκτέλεση hybrid search
        results = self.search(query, n_results=6)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Δημιουργία context
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results):
            # Μορφοποίηση κάθε Q&A pair
            qa_text = f"Q: {result.qa_pair.question}\nA: {result.qa_pair.answer}"
            
            # Έλεγχος αν χωράει στο context
            if total_length + len(qa_text) > max_context_length:
                break
            
            context_parts.append(qa_text)
            total_length += len(qa_text)
        
        # Συνδυασμός με διαχωριστικά
        context = "\n\n---\n\n".join(context_parts)
        
        # Προσθήκη header για το LLM
        header = f"Here are the most relevant Q&A pairs for the query '{query}':\n\n"
        
        return header + context
    
    def explain_results(self, query: str) -> Dict:
        """
        Εξηγεί πώς λειτούργησε το hybrid search για μια ερώτηση.
        
        Χρήσιμο για debugging και για να καταλάβουμε γιατί
        επιλέχθηκαν συγκεκριμένα αποτελέσματα.
        
        Args:
            query: Η ερώτηση για ανάλυση
            
        Returns:
            Dictionary με detailed explanation
        """
        results = self.search(query)
        
        # Ανάλυση των αποτελεσμάτων
        semantic_only = [r for r in results if r.match_type == "semantic"]
        keyword_only = [r for r in results if r.match_type == "keyword"]  
        both_match = [r for r in results if r.match_type == "both"]
        
        # Σημαντικοί όροι από TF-IDF
        important_terms = self.tfidf_service.get_important_terms(query, top_n=5)
        
        return {
            "query": query,
            "total_results": len(results),
            "semantic_only": len(semantic_only),
            "keyword_only": len(keyword_only),
            "both_match": len(both_match),
            "top_result": {
                "question": results[0].qa_pair.question if results else None,
                "score": results[0].combined_score if results else 0,
                "type": results[0].match_type if results else None
            },
            "important_keywords": important_terms,
            "weights": {
                "semantic": self.semantic_weight,
                "keyword": self.keyword_weight
            }
        }