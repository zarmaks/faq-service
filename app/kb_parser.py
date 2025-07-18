"""
Knowledge Base Parser - Διαβάζει και οργανώνει το knowledge base.

Αυτό το module είναι υπεύθυνο για να μετατρέψει το raw text file
σε δομημένα Q&A pairs που μπορούμε να επεξεργαστούμε.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """
    Αναπαριστά ένα ζευγάρι ερώτησης-απάντησης.
    
    Χρησιμοποιούμε dataclass για clean και type-safe κώδικα.
    Κάθε QAPair είναι ένα αυτόνομο chunk πληροφορίας.
    """
    question: str
    answer: str
    id: int  # Μοναδικό ID για κάθε pair
    
    def to_text(self) -> str:
        """
        Μετατρέπει το Q&A pair σε ένα ενιαίο κείμενο για indexing.
        
        Συνδυάζουμε question και answer γιατί θέλουμε το search
        να βρίσκει chunks είτε από την ερώτηση είτε από την απάντηση.
        """
        return f"Question: {self.question}\nAnswer: {self.answer}"
    
    def to_dict(self) -> Dict[str, any]:
        """Μετατροπή σε dictionary για εύκολη αποθήκευση."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "full_text": self.to_text()
        }


class KnowledgeBaseParser:
    """
    Parser για το knowledge base file.
    
    Διαβάζει το text file και εξάγει δομημένα Q&A pairs.
    Η λογική parsing βασίζεται στο format: Q: ... A: ...
    """
    
    def __init__(self, file_path: str):
        """
        Initialize με το path του knowledge base file.
        
        Args:
            file_path: Path στο knowledge base text file
        """
        self.file_path = file_path
        self.qa_pairs: List[QAPair] = []
        
    def parse(self) -> List[QAPair]:
        """
        Διαβάζει και αναλύει το knowledge base file.
        
        Η διαδικασία:
        1. Διαβάζει όλο το αρχείο
        2. Χρησιμοποιεί regex για να βρει Q: και A: patterns
        3. Δημιουργεί QAPair objects
        4. Καθαρίζει και validates τα δεδομένα
        
        Returns:
            List of QAPair objects
        """
        try:
            # Διαβάζουμε το αρχείο
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Regex pattern για να βρούμε Q&A pairs
            # Ψάχνουμε για "Q:" στην αρχή γραμμής, μετά οτιδήποτε μέχρι "A:", 
            # και μετά την απάντηση μέχρι το επόμενο "Q:" ή το τέλος
            pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'
            
            # Βρίσκουμε όλα τα matches
            matches = re.findall(pattern, content, re.DOTALL)
            
            # Δημιουργούμε QAPair objects
            self.qa_pairs = []
            for i, (question, answer) in enumerate(matches):
                # Καθαρίζουμε whitespace και empty lines
                question = self._clean_text(question)
                answer = self._clean_text(answer)
                
                # Μόνο αν έχουμε και question και answer
                if question and answer:
                    qa_pair = QAPair(
                        id=i,
                        question=question,
                        answer=answer
                    )
                    self.qa_pairs.append(qa_pair)
            
            logger.info(f"✅ Parsed {len(self.qa_pairs)} Q&A pairs from knowledge base")
            
            # Validation - τουλάχιστον κάποια pairs πρέπει να βρεθούν
            if not self.qa_pairs:
                raise ValueError("No Q&A pairs found in knowledge base!")
            
            return self.qa_pairs
            
        except FileNotFoundError:
            logger.error(f"Knowledge base file not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error parsing knowledge base: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Καθαρίζει το κείμενο από περιττά whitespaces και characters.
        
        Args:
            text: Raw text από το file
            
        Returns:
            Καθαρισμένο κείμενο
        """
        # Αφαιρούμε περιττά whitespaces
        text = ' '.join(text.split())
        
        # Αφαιρούμε leading/trailing whitespace
        text = text.strip()
        
        # Αντικαθιστούμε πολλαπλά spaces με ένα
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def get_by_id(self, qa_id: int) -> QAPair:
        """
        Επιστρέφει ένα συγκεκριμένο QAPair βάσει ID.
        
        Χρήσιμο όταν το search επιστρέφει IDs και θέλουμε
        να ανακτήσουμε το πλήρες content.
        """
        for qa in self.qa_pairs:
            if qa.id == qa_id:
                return qa
        raise ValueError(f"QAPair with id {qa_id} not found")
    
    def search_by_keyword(self, keyword: str) -> List[QAPair]:
        """
        Απλή keyword search στα Q&A pairs.
        
        Αυτό είναι ένα fallback για πολύ απλές αναζητήσεις.
        Το κύριο search θα γίνεται με ChromaDB και TF-IDF.
        """
        keyword_lower = keyword.lower()
        results = []
        
        for qa in self.qa_pairs:
            if (keyword_lower in qa.question.lower() or 
                keyword_lower in qa.answer.lower()):
                results.append(qa)
        
        return results
    
    def get_stats(self) -> Dict[str, any]:
        """
        Επιστρέφει στατιστικά για το knowledge base.
        
        Χρήσιμο για debugging και monitoring.
        """
        if not self.qa_pairs:
            return {"error": "No Q&A pairs loaded"}
        
        total_chars = sum(len(qa.to_text()) for qa in self.qa_pairs)
        avg_q_length = sum(len(qa.question) for qa in self.qa_pairs) / len(self.qa_pairs)
        avg_a_length = sum(len(qa.answer) for qa in self.qa_pairs) / len(self.qa_pairs)
        
        return {
            "total_pairs": len(self.qa_pairs),
            "total_characters": total_chars,
            "average_question_length": round(avg_q_length),
            "average_answer_length": round(avg_a_length),
            "shortest_question": min(self.qa_pairs, key=lambda x: len(x.question)).question[:50],
            "longest_answer_preview": max(self.qa_pairs, key=lambda x: len(x.answer)).answer[:100] + "..."
        }


# Utility function για γρήγορη χρήση
def load_knowledge_base(file_path: str) -> List[QAPair]:
    """
    Shortcut function για να φορτώσεις το knowledge base.
    
    Usage:
        qa_pairs = load_knowledge_base("data/knowledge_base.txt")
    """
    parser = KnowledgeBaseParser(file_path)
    return parser.parse()