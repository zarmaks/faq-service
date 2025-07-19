#!/usr/bin/env python3
"""
Test to prove the embedding dilution problem
"""
import sys
sys.path.append('.')
from app.embeddings_service import EmbeddingsService

def main():
    emb_service = EmbeddingsService()

    # Test query
    query = 'SOC 2 compliance certification'
    
    # Test 1: Μόνο η ερώτηση (όπως στο test_embeddings)
    question_only = 'What industry compliance certifications do you have?'
    
    # Test 2: Το συνδυασμένο κείμενο (όπως στο ChromaDB)
    combined = '''Question: What industry compliance certifications do you have?
Answer: CloudSphere is SOC 2 Type II, ISO 27001, GDPR, and HIPAA-ready. Formal audit reports can be requested by emailing compliance@cloudsphere.com from an authorised corporate domain.'''

    # Test 3: Μόνο η απάντηση
    answer_only = 'CloudSphere is SOC 2 Type II, ISO 27001, GDPR, and HIPAA-ready. Formal audit reports can be requested by emailing compliance@cloudsphere.com from an authorised corporate domain.'

    print(f'🔍 Query: "{query}"')
    print()
    
    query_emb = emb_service.create_embedding(query)
    
    # Test όλες τις παραλλαγές
    tests = [
        ("Question only", question_only),
        ("Answer only", answer_only), 
        ("Combined Q+A", combined)
    ]
    
    for name, text in tests:
        text_emb = emb_service.create_embedding(text)
        similarity = emb_service.cosine_similarity(query_emb, text_emb)
        print(f'{name:15} similarity: {similarity:.6f}')
        print(f'{"":15} Above 0.05? {similarity > 0.05}')
        print()

    # BONUS: Ελέγχουμε τι έχει αποθηκευτεί στο ChromaDB
    print("="*50)
    print("🔍 CHECKING CHROMADB STORED EMBEDDINGS")
    print("="*50)
    
    from app.chromadb_service import ChromaDBService
    chromadb = ChromaDBService()
    
    # Παίρνουμε τα stored documents
    result = chromadb.collection.get()
    soc_docs = [doc for doc in result['documents'] if 'SOC' in doc]
    
    if soc_docs:
        stored_doc = soc_docs[0]
        print(f"Stored document preview: {stored_doc[:100]}...")
        
        # Υπολογίζουμε manual similarity
        stored_emb = emb_service.create_embedding(stored_doc)
        manual_sim = emb_service.cosine_similarity(query_emb, stored_emb)
        
        # Σύγκριση με ChromaDB search
        search_results = chromadb.search(query_emb, n_results=1)
        chromadb_sim = search_results[0].similarity
        
        print(f"Manual calculation: {manual_sim:.6f}")
        print(f"ChromaDB search:    {chromadb_sim:.6f}")
        print(f"Difference:         {abs(manual_sim - chromadb_sim):.6f}")
        
        if abs(manual_sim - chromadb_sim) > 0.1:
            print("⚠️  BIG DIFFERENCE! Possible ChromaDB issue!")
        else:
            print("✅ ChromaDB calculation matches manual")


if __name__ == "__main__":
    main()
