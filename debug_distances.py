#!/usr/bin/env python3
"""
Quick debug για να δούμε τα πραγματικά ChromaDB distances.
"""

import sys
sys.path.append('.')
from app.chromadb_service import ChromaDBService
from app.embeddings_service import EmbeddingsService
import numpy as np

print('🔍 DEBUGGING CHROMADB DISTANCES')
print('='*50)

try:
    embeddings_service = EmbeddingsService() 
    chromadb_service = ChromaDBService()
    
    # Test queries
    queries = [
        'SOC 2 compliance certification',
        'What is your refund policy?',
        'Can I deploy with Docker?'
    ]
    
    for query in queries:
        print(f'\n🔎 Query: "{query}"')
        
        # Get embedding
        query_embedding = embeddings_service.create_embedding(query)
        query_norm = np.linalg.norm(query_embedding)
        print(f'Query embedding norm: {query_norm:.2f}')
        
        # Raw ChromaDB query
        results = chromadb_service.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        distances = results['distances'][0]
        documents = results['documents'][0]
        
        for i, distance in enumerate(distances):
            old_sim = max(0.0, 1.0 - (distance / 400.0))
            new_sim = 1.0 / (1.0 + distance)
            
            print(f'  Result {i+1}:')
            print(f'    Distance: {distance:.6f}')
            print(f'    Old similarity: {old_sim:.6f}')  
            print(f'    New similarity: {new_sim:.6f}')
            print(f'    Doc preview: {documents[i][:60]}...')
            print()

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
