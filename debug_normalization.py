#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath('.'))

from app.embeddings_service import EmbeddingsService

def debug_embedding_normalization():
    print("🔍 Debugging embedding normalization...")
    
    embeddings_service = EmbeddingsService()
    
    # Test text
    text = "SOC 2 compliance certification"
    
    # Get embedding
    embedding = embeddings_service.create_embedding(text)
    embedding_array = np.array(embedding)
    
    print(f"Text: '{text}'")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"Embedding magnitude (L2 norm): {np.linalg.norm(embedding_array):.6f}")
    print(f"Min value: {np.min(embedding_array):.6f}")
    print(f"Max value: {np.max(embedding_array):.6f}")
    print(f"Mean value: {np.mean(embedding_array):.6f}")
    print(f"Standard deviation: {np.std(embedding_array):.6f}")
    
    # Check if it's normalized
    norm = np.linalg.norm(embedding_array)
    is_normalized = abs(norm - 1.0) < 0.01
    print(f"Is normalized (norm ≈ 1.0): {is_normalized}")
    
    if not is_normalized:
        print("\n⚠️  Embeddings are NOT normalized!")
        print("This explains the large L2 distances in ChromaDB")
        
        # Show what normalized would look like
        normalized = embedding_array / norm
        print(f"Normalized magnitude: {np.linalg.norm(normalized):.6f}")

if __name__ == "__main__":
    debug_embedding_normalization()
