"""
Test script για τα embeddings.

Τρέξε το με: python test_embeddings.py
"""

from app.embeddings_service import EmbeddingsService

def main():
    print("🚀 Testing Embeddings Service\n")
    
    # Δημιουργούμε την service
    service = EmbeddingsService()
    
    # Test 1: Δημιουργία ενός embedding
    print("📝 Test 1: Creating a single embedding")
    text = "What is the refund policy?"
    embedding = service.create_embedding(text)
    print(f"Text: '{text}'")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Το embedding είναι ένας πίνακας με {len(embedding)} αριθμούς!\n")
    
    # Test 2: Σύγκριση similarities
    print("📊 Test 2: Comparing text similarities")
    service.test_similarity()
    
    # Test 3: Batch embeddings
    print("🔄 Test 3: Creating batch embeddings")
    texts = [
        "How do I reset my password?",
        "What payment methods do you accept?",
        "Can I get a refund?",
        "How secure is my data?"
    ]
    
    embeddings = service.create_embeddings_batch(texts)
    print(f"\n✅ Created {len(embeddings)} embeddings")
    
    # Bonus: Δείξε πώς παρόμοια κείμενα έχουν κοντινά embeddings
    print("\n🎯 Bonus: Finding similar texts")
    query = "I forgot my password"
    query_emb = service.create_embedding(query)
    
    print(f"\nQuery: '{query}'")
    print("Similarities with other texts:")
    
    for i, text in enumerate(texts):
        similarity = service.cosine_similarity(query_emb, embeddings[i])
        print(f"  - '{text}': {similarity:.3f}")

if __name__ == "__main__":
    main()