#!/usr/bin/env python3
"""
Test για τη διόρθωση του distance-to-similarity conversion.
"""

import numpy as np

def test_distance_similarity_conversion():
    """Δοκιμάζει διαφορετικές μεθόδους μετατροπής distance σε similarity."""
    
    print("🧮 Testing Distance to Similarity Conversion")
    print("=" * 50)
    
    # Simulated L2 distances from ChromaDB (based on our embeddings)
    test_distances = [
        0.0,   # Πανομοιότυπα
        5.0,   # Πολύ παρόμοια  
        10.0,  # Αρκετά παρόμοια
        15.0,  # Κάπως παρόμοια
        20.0,  # Λίγο παρόμοια
        30.0,  # Άσχετα
        50.0   # Εντελώς άσχετα
    ]
    
    print(f"{'Distance':<10} {'Old Method':<12} {'New Method':<12} {'Exp Method':<12}")
    print("-" * 50)
    
    for distance in test_distances:
        # Παλιά μέθοδος (λάθος)
        old_similarity = max(0.0, 1.0 - (distance / 400.0))
        
        # Νέα μέθοδος (inverse)  
        new_similarity = 1.0 / (1.0 + distance)
        
        # Εναλλακτική μέθοδος (exponential)
        exp_similarity = np.exp(-distance / 10.0)
        
        print(f"{distance:<10.1f} {old_similarity:<12.3f} {new_similarity:<12.3f} {exp_similarity:<12.3f}")
    
    print("\n💡 Analysis:")
    print("- Old method: Gives negative values for distances > 400")
    print("- New method (inverse): Always positive, good range 0-1")
    print("- Exp method: Similar to inverse but with exponential decay")
    
    # Test με real-world παραδείγματα
    print("\n🎯 Real-world examples:")
    examples = [
        (0.5, "Same document, tiny numerical differences"),
        (8.0, "Very similar semantically"),
        (15.0, "Somewhat related"),
        (25.0, "Weakly related"),
        (40.0, "Mostly unrelated")
    ]
    
    for distance, description in examples:
        similarity = 1.0 / (1.0 + distance)
        print(f"Distance {distance:4.1f} → Similarity {similarity:.3f} | {description}")

def test_cosine_vs_l2():
    """Δείχνει τη διαφορά μεταξύ cosine και L2 distance."""
    print("\n🔍 Cosine vs L2 Distance Comparison")
    print("=" * 50)
    
    # Δύο embeddings με διαφορετικά norms
    emb1 = np.array([1.0, 2.0, 3.0]) * 10  # norm ~37.4
    emb2 = np.array([1.1, 1.9, 3.1]) * 10  # norm ~37.2, αλλά παρόμοιο
    emb3 = np.array([3.0, 1.0, 2.0]) * 10  # norm ~37.4, διαφορετικό
    
    pairs = [
        ("Similar vectors", emb1, emb2),
        ("Different vectors", emb1, emb3)
    ]
    
    for name, vec1, vec2 in pairs:
        # L2 distance
        l2_dist = np.linalg.norm(vec1 - vec2)
        
        # Cosine similarity
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Our conversion
        our_similarity = 1.0 / (1.0 + l2_dist)
        
        print(f"{name}:")
        print(f"  L2 distance:     {l2_dist:.3f}")
        print(f"  Cosine similarity: {cosine_sim:.3f}")
        print(f"  Our similarity:  {our_similarity:.3f}")
        print()

if __name__ == "__main__":
    test_distance_similarity_conversion()
    test_cosine_vs_l2()
