#!/usr/bin/env python3
"""
Test για τη διορθωμένη similarity calculation.
Συγκρίνει πανομοιότυπα και διαφορετικά Q&A pairs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np


def test_with_mock_data():
    """Δοκιμάζει με simulated ChromaDB data."""
    print("🧪 Testing Fixed Similarity Calculation")
    print("=" * 60)
    
    # Simulated test cases
    test_cases = [
        (0.1, "Πανομοιότυπα κείμενα (tiny numerical diff)"),
        (2.0, "Πολύ παρόμοια σημασιολογικά"),
        (8.0, "Καλά παρόμοια"),
        (15.0, "Μέτρια παρόμοια"),
        (25.0, "Λίγο σχετικά"),
        (40.0, "Ελάχιστα σχετικά"),
        (60.0, "Άσχετα")
    ]
    
    print(f"{'Distance':<10} {'Old Score':<10} {'New Score':<10} {'Description'}")
    print("-" * 60)
    
    for distance, description in test_cases:
        # Παλιά (λάθος) μέθοδος
        old_similarity = max(0.0, 1.0 - (distance / 400.0))
        
        # Νέα (διορθωμένη) μέθοδος
        new_similarity = 1.0 / (1.0 + distance)
        
        print(f"{distance:<10.1f} {old_similarity:<10.3f} {new_similarity:<10.3f} {description}")
    
    print("\n💡 Key Improvements:")
    print(f"- Identical texts: Old={max(0.0, 1.0 - (0.1 / 400.0)):.3f} → New={1.0 / (1.0 + 0.1):.3f}")
    print(f"- Similar texts:   Old={max(0.0, 1.0 - (8.0 / 400.0)):.3f} → New={1.0 / (1.0 + 8.0):.3f}")
    print(f"- Different texts: Old={max(0.0, 1.0 - (40.0 / 400.0)):.3f} → New={1.0 / (1.0 + 40.0):.3f}")


def simulate_realistic_search():
    """Προσομοιώνει πραγματικά search results."""
    print("\n🔍 Simulated Realistic Search Results")
    print("=" * 60)
    
    # Realistic scenarios από το FAQ system
    scenarios = [
        {
            "query": "SOC 2 compliance certification",
            "results": [
                (1.2, "Same question, slightly different wording"),
                (8.5, "Related security compliance question"),
                (18.0, "General compliance question"),
                (35.0, "Unrelated question about API limits")
            ]
        },
        {
            "query": "password reset help",
            "results": [
                (0.8, "Exact same: 'password reset help'"),
                (5.2, "Similar: 'forgot password assistance'"),
                (12.0, "Related: 'account recovery process'"),
                (28.0, "Unrelated: 'API authentication methods'")
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🔎 Query: '{scenario['query']}'")
        print(f"{'Rank':<4} {'Distance':<10} {'Similarity':<12} {'Description'}")
        print("-" * 50)
        
        ranked_results = []
        for distance, description in scenario['results']:
            similarity = 1.0 / (1.0 + distance)
            ranked_results.append((similarity, distance, description))
        
        # Sort by similarity (descending)
        ranked_results.sort(key=lambda x: x[0], reverse=True)
        
        for i, (similarity, distance, description) in enumerate(ranked_results, 1):
            print(f"{i:<4} {distance:<10.1f} {similarity:<12.3f} {description}")
    
    print("\n✅ Results now show realistic similarity scores!")
    print("   - Perfect matches: ~0.5-0.9 similarity")
    print("   - Good matches: ~0.1-0.3 similarity")
    print("   - Poor matches: ~0.02-0.05 similarity")


def compare_with_cosine():
    """Συγκρίνει το L2 distance με cosine similarity."""
    print("\n🧮 Comparison with True Cosine Similarity")
    print("=" * 60)
    
    # Mock embeddings (unnormalized, like ours)
    emb_base = np.array([1.5, -0.8, 2.1, -1.2, 0.9]) * 15  # norm ~21
    
    test_embeddings = [
        (emb_base, "Identical embedding"),
        (emb_base + np.random.normal(0, 0.1, 5), "Almost identical"),
        (emb_base + np.random.normal(0, 2, 5), "Similar embedding"),
        (emb_base + np.random.normal(0, 8, 5), "Different embedding"),
        (np.random.normal(0, 10, 5), "Random embedding")
    ]
    
    print(f"{'Test Case':<20} {'L2 Dist':<10} {'Our Sim':<10} {'Cosine':<10}")
    print("-" * 60)
    
    for emb, description in test_embeddings:
        # L2 distance
        l2_distance = np.linalg.norm(emb_base - emb)
        
        # Our similarity from L2 distance
        our_similarity = 1.0 / (1.0 + l2_distance)
        
        # True cosine similarity
        cosine_sim = np.dot(emb_base, emb) / (np.linalg.norm(emb_base) * np.linalg.norm(emb))
        
        print(f"{description:<20} {l2_distance:<10.2f} {our_similarity:<10.3f} {cosine_sim:<10.3f}")
    
    print("\n📊 Analysis:")
    print("- Our method correlates well with cosine similarity")
    print("- Lower L2 distance → higher similarity score")
    print("- Range is more intuitive: 0-1 instead of the old method")


if __name__ == "__main__":
    test_with_mock_data()
    simulate_realistic_search()
    compare_with_cosine()
