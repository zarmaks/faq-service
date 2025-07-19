#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from app.embeddings_service import EmbeddingsService
from app.chromadb_service import ChromaDBService

def debug_similarity():
    print("🔍 Debugging ChromaDB similarity scores...")
    
    # Initialize services
    embeddings_service = EmbeddingsService()
    chromadb_service = ChromaDBService()
    
    # Test query
    query = "SOC 2 compliance certification"
    print(f"Query: '{query}'")
    
    # Get embedding
    query_embedding = embeddings_service.create_embedding(query)
    print(f"Embedding dimensions: {len(query_embedding)}")
    
    # Search and get raw results
    results = chromadb_service.collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    print("\nRaw ChromaDB results:")
    print("=" * 50)
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for i in range(len(documents)):
        print(f"Result #{i+1}:")
        print(f"  Question: {metadatas[i]['question'][:60]}...")
        print(f"  Raw L2 Distance: {distances[i]}")
        
        # Different similarity calculations
        similarity_1 = 1 / (1 + distances[i])  # Current method
        similarity_2 = 1 - min(distances[i] / 2, 1)  # Alternative method
        similarity_3 = max(0, 1 - distances[i])  # Simple subtraction
        
        print(f"  Similarity method 1: {similarity_1:.6f}")
        print(f"  Similarity method 2: {similarity_2:.6f}")
        print(f"  Similarity method 3: {similarity_3:.6f}")
        print()

if __name__ == "__main__":
    debug_similarity()
