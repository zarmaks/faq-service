"""
Test script για να δούμε πώς λειτουργεί το cosine similarity.

Τρέξε το με: python test_cosine_similarity.py
"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"


def test_similarity_scores():
    """Δοκιμάζει διάφορες ερωτήσεις για να δούμε τα similarity scores."""
    
    print("🧪 Testing Cosine Similarity Scores\n")
    print("=" * 70)
    
    # Δοκιμαστικές ερωτήσεις - από πολύ σχετικές μέχρι άσχετες
    test_cases = [
        {
            "query": "What is the refund policy?",
            "expected": "Should find exact match with very high score"
        },
        {
            "query": "How can I get my money back?",
            "expected": "Should find refund policy with high score (semantic match)"
        },
        {
            "query": "I want a refund for my purchase",
            "expected": "Should find refund policy with good score"
        },
        {
            "query": "Tell me about returns and refunds",
            "expected": "Should find refund policy with good score"
        },
        {
            "query": "What are your security certifications?",
            "expected": "Should find SOC 2/compliance info"
        },
        {
            "query": "Is my data safe?",
            "expected": "Should find security-related Q&As"
        },
        {
            "query": "How much does it cost?",
            "expected": "Should find pricing information"
        },
        {
            "query": "The weather is nice today",
            "expected": "Should have low scores - unrelated to FAQ"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {test['query']}")
        print(f"Expected: {test['expected']}")
        print("-" * 70)
        
        # Κάνουμε το RAG search request
        response = requests.post(
            f"{BASE_URL}/rag/search",
            json={"question": test['query']}
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data['results']
            
            if not results:
                print("❌ No results found!")
                continue
            
            print(f"\nTop 3 Results:")
            for j, result in enumerate(results[:3]):
                similarity = result['scores']['combined']
                semantic = result['scores']['semantic']
                keyword = result['scores']['keyword']
                
                # Χρωματισμός βάσει score
                if similarity > 0.8:
                    score_indicator = "🟢"  # Excellent match
                elif similarity > 0.6:
                    score_indicator = "🟡"  # Good match
                elif similarity > 0.4:
                    score_indicator = "🟠"  # Moderate match
                else:
                    score_indicator = "🔴"  # Poor match
                
                print(f"\n  {j+1}. {score_indicator} Combined Score: {similarity:.3f}")
                print(f"     Question: {result['question'][:60]}...")
                print(f"     Semantic: {semantic:.3f} | Keyword: {keyword:.3f}")
                print(f"     Match Type: {result['match_type']}")
            
            # Ανάλυση αποτελεσμάτων
            print(f"\nAnalysis:")
            explanation = data['explanation']
            print(f"  - Total results: {explanation['total_results']}")
            print(f"  - Semantic only: {explanation['semantic_only']}")
            print(f"  - Keyword only: {explanation['keyword_only']}")
            print(f"  - Both methods: {explanation['both_match']}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
    
    print(f"\n{'='*70}")
    print("\n📊 Score Legend:")
    print("  🟢 > 0.8  : Excellent match")
    print("  🟡 > 0.6  : Good match")
    print("  🟠 > 0.4  : Moderate match")
    print("  🔴 ≤ 0.4  : Poor match")


def test_specific_comparison():
    """Συγκρίνει συγκεκριμένες ερωτήσεις για να δούμε τη διαφορά."""
    
    print("\n\n🔬 Detailed Comparison Test")
    print("=" * 70)
    
    # Η βασική ερώτηση
    base_query = "What is the refund policy?"
    
    # Παραλλαγές
    variations = [
        "What is the refund policy?",  # Exact match
        "refund policy",                # Keywords only
        "How do I get a refund?",       # Semantic variation
        "Can I return my purchase?",    # Different words, same concept
        "Money back guarantee?",        # Very different words
        "What is CloudSphere?",         # Different topic
    ]
    
    print(f"\nBase Query: '{base_query}'")
    print("\nComparing with variations:")
    
    for variation in variations:
        response = requests.post(
            f"{BASE_URL}/rag/search",
            json={"question": variation}
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data['results']
            
            # Βρίσκουμε το refund policy Q&A στα αποτελέσματα
            refund_result = None
            for result in results:
                if "refund" in result['question'].lower():
                    refund_result = result
                    break
            
            if refund_result:
                similarity = refund_result['scores']['combined']
                print(f"\n  '{variation}'")
                print(f"    → Similarity: {similarity:.3f}")
            else:
                print(f"\n  '{variation}'")
                print(f"    → Not in top results (too dissimilar)")


if __name__ == "__main__":
    print("🚀 Starting Cosine Similarity Tests")
    print("Make sure the API is running with the updated ChromaDB service\n")
    
    test_similarity_scores()
    test_specific_comparison()
    
    print("\n\n✅ Testing complete!")
    print("\nIf the scores look more intuitive now (similar questions getting")
    print("higher scores), then the cosine distance is working better than L2!")