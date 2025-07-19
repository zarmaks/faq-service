"""
TF-IDF Service - Keyword-based search.

Το TF-IDF βρίσκει έγγραφα βάσει συγκεκριμένων λέξεων-κλειδιών.
Είναι το συμπλήρωμα του semantic search - πιάνει ακριβείς όρους
που μπορεί να χάσει το semantic search.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class TFIDFService:
    """
    Service για keyword-based search με TF-IDF.
    
    Το TF-IDF είναι ιδανικό για να βρούμε έγγραφα που περιέχουν
    συγκεκριμένους όρους, ακρωνύμια, ή τεχνικές λέξεις.
    """
    
    def __init__(self):
        """Initialize TF-IDF service."""
        # Ο TfidfVectorizer μετατρέπει κείμενα σε TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            # Χρησιμοποιούμε 1-3 grams (μονές λέξεις, ζευγάρια, τριάδες)
            ngram_range=(1, 3),
            
            # Αγνοούμε πολύ κοινές λέξεις (the, is, at, etc.)
            stop_words='english',
            
            # Κρατάμε max 1000 features για performance
            max_features=1000,
            
            # Αγνοούμε λέξεις που εμφανίζονται σε >80% των εγγράφων
            max_df=0.8,
            
            # Αγνοούμε λέξεις που εμφανίζονται σε <2 έγγραφα
            min_df=2,
            
            # Χρησιμοποιούμε sublinear scaling για καλύτερα αποτελέσματα
            sublinear_tf=True
        )
        
        self.documents = []
        self.document_vectors = None
        self.is_fitted = False
        
    def preprocess_text(self, text: str) -> str:
        """
        Προεπεξεργασία κειμένου για TF-IDF.
        
        Κάνουμε:
        - Lowercase για case-insensitive matching
        - Αφαίρεση περιττών χαρακτήρων
        - Διατήρηση σημαντικών στοιχείων (emails, URLs, numbers)
        
        Args:
            text: Το αρχικό κείμενο
            
        Returns:
            Καθαρισμένο κείμενο
        """
        # Lowercase
        text = text.lower()
        
        # Διατηρούμε emails ως ενιαίες λέξεις
        text = re.sub(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', 
                     lambda m: m.group(0).replace('.', 'DOT').replace('@', 'AT'), 
                     text)
        
        # Διατηρούμε URLs
        text = re.sub(r'(https?://[^\s]+)', 
                     lambda m: m.group(0).replace('/', 'SLASH').replace('.', 'DOT'), 
                     text)
        
        # Αφαιρούμε ειδικούς χαρακτήρες αλλά κρατάμε αριθμούς και -
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
        
        # Πολλαπλά spaces σε ένα
        text = ' '.join(text.split())
        
        return text
    
    def fit(self, documents: List[str], qa_pairs: List[Dict]):
        """
        "Εκπαιδεύει" το TF-IDF model με τα έγγραφα.
        
        Υπολογίζει τις συχνότητες των λέξεων και δημιουργεί
        τους TF-IDF vectors για κάθε έγγραφο.
        
        Args:
            documents: List με τα κείμενα των Q&A pairs
            qa_pairs: List με τα αντίστοιχα Q&A objects
        """
        # Προεπεξεργασία κειμένων
        self.documents = [self.preprocess_text(doc) for doc in documents]
        self.qa_pairs = qa_pairs
        
        # Εκπαίδευση του vectorizer και μετατροπή εγγράφων
        try:
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
            self.is_fitted = True
            
            # Logging για debugging
            feature_names = self.vectorizer.get_feature_names_out()
            logger.info(f"✅ TF-IDF fitted with {len(self.documents)} documents")
            logger.info(f"   Vocabulary size: {len(feature_names)}")
            logger.info(f"   Sample features: {list(feature_names[:10])}")
            
        except ValueError as e:
            logger.error(f"Error fitting TF-IDF: {e}")
            logger.error("This usually happens with too few documents or all stop words")
            # Fallback σε απλούστερο vectorizer
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
            self.is_fitted = True
    
    def search(self, query: str, n_results: int = 3) -> List[Tuple[int, float]]:
        """
        Αναζήτηση με TF-IDF.
        
        Βρίσκει τα έγγραφα με τις πιο σχετικές λέξεις-κλειδιά.
        
        Args:
            query: Η ερώτηση του χρήστη
            n_results: Πόσα αποτελέσματα να επιστρέψει
            
        Returns:
            List of tuples (qa_id, similarity_score)
        """
        if not self.is_fitted:
            logger.error("TF-IDF not fitted yet!")
            return []
        
        # Προεπεξεργασία query
        processed_query = self.preprocess_text(query)
        
        # Μετατροπή query σε TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Υπολογισμός cosine similarity με όλα τα έγγραφα
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Βρίσκουμε τα top-n αποτελέσματα
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        # Δημιουργούμε τα αποτελέσματα
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Μόνο αν υπάρχει κάποια ομοιότητα
                qa_id = self.qa_pairs[idx]['id']
                score = float(similarities[idx])
                results.append((qa_id, score))
        
        return results
    
    def get_important_terms(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Βρίσκει τους πιο σημαντικούς όρους σε ένα κείμενο.
        
        Χρήσιμο για debugging και για να καταλάβουμε
        ποιες λέξεις θεωρεί σημαντικές το TF-IDF.
        
        Args:
            text: Το κείμενο για ανάλυση
            top_n: Πόσους όρους να επιστρέψει
            
        Returns:
            List of tuples (term, tfidf_score)
        """
        if not self.is_fitted:
            return []
        
        # Προεπεξεργασία και μετατροπή
        processed_text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([processed_text])
        
        # Παίρνουμε τα feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Παίρνουμε τα TF-IDF scores
        tfidf_scores = text_vector.toarray()[0]
        
        # Βρίσκουμε non-zero scores
        non_zero_indices = np.where(tfidf_scores > 0)[0]
        
        # Δημιουργούμε list με (term, score)
        term_scores = [
            (feature_names[i], tfidf_scores[i]) 
            for i in non_zero_indices
        ]
        
        # Ταξινομούμε κατά score
        term_scores.sort(key=lambda x: x[1], reverse=True)
        
        return term_scores[:top_n]
    
    def explain_search(self, query: str, result_idx: int) -> Dict:
        """
        Εξηγεί γιατί ένα αποτέλεσμα ταιριάζει με το query.
        
        Δείχνει ποιες λέξεις-κλειδιά ταίριαξαν.
        
        Args:
            query: Η ερώτηση
            result_idx: Ο index του αποτελέσματος
            
        Returns:
            Dictionary με την εξήγηση
        """
        if not self.is_fitted or result_idx >= len(self.documents):
            return {"error": "Invalid result index"}
        
        # Terms από το query
        query_terms = set(self.preprocess_text(query).split())
        
        # Terms από το αποτέλεσμα
        result_terms = set(self.documents[result_idx].split())
        
        # Κοινά terms
        matching_terms = query_terms.intersection(result_terms)
        
        # TF-IDF scores για τα matching terms
        important_query_terms = self.get_important_terms(query, top_n=10)
        important_result_terms = self.get_important_terms(
            self.documents[result_idx], 
            top_n=10
        )
        
        return {
            "matching_keywords": list(matching_terms),
            "query_important_terms": important_query_terms[:5],
            "result_important_terms": important_result_terms[:5],
            "explanation": f"Matched {len(matching_terms)} keywords"
        }