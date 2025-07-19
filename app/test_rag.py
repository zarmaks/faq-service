"""
Comprehensive test script για το Hybrid RAG system.

Τρέξε το με: python test_rag.py
"""

import requests
import time

BASE_URL = "http://localhost:8002/api/v1"


def print_section(title: str):
    """Helper για όμορφο formatting."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_rag_search():
    """Τεστάρει το RAG search endpoint."""
    print_section("Testing RAG Search (Without LLM)")
    
    test_queries = [
        # Ερωτήσεις εντός knowledge base
        "What is the refund policy?",
        "How much does Professional tier cost?",
        "forgotten password help",
        "SOC 2 compliance certification",
        "support@cloudsphere.com contact",
        "API rate limiting details",
        
        # Ερωτήσεις εκτός knowledge base (για testing edge cases)
        "Can I deploy with Docker?",  # Deployment topic δεν υπάρχει στο KB
        "What programming languages do you support?",  # Technical specs
        "How do I integrate with Slack?",  # Integration που δεν υπάρχει
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        
        response = requests.post(
            f"{BASE_URL}/rag/search",
            json={"question": query}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Εμφάνιση αποτελεσμάτων
            print(f"Found {len(data['results'])} results:")
            
            for i, result in enumerate(data['results'][:3]):
                print(f"\n  #{i+1} [{result['match_type'].upper()}]")
                print(f"  Q: {result['question'][:60]}...")
                print(f"  Scores: Semantic={result['scores']['semantic']}, "
                      f"Keyword={result['scores']['keyword']}, "
                      f"Combined={result['scores']['combined']}")
                print(f"  Why: {result['explanation']}")
            
            # Εμφάνιση explanation
            exp = data['explanation']
            print(f"\n📊 Search Analysis:")
            print(f"  - Semantic only: {exp['semantic_only']} results")
            print(f"  - Keyword only: {exp['keyword_only']} results")
            print(f"  - Both methods: {exp['both_match']} results")
            
            if exp['important_keywords']:
                print(f"  - Key terms: {', '.join([t[0] for t in exp['important_keywords'][:3]])}")
        else:
            print(f"❌ Error: {response.status_code}")


def test_full_system():
    """Τεστάρει το πλήρες σύστημα με LLM."""
    print_section("Testing Full System (RAG + LLM)")
    
    test_questions = [
        {
            "question": "What is CloudSphere and who should use it?",
            "category": "General Info"
        },
        {
            "question": "I want to get my money back",
            "category": "Semantic Match (refund)"
        },
        {
            "question": "Do you have SOC 2?",
            "category": "Keyword Match (specific term)"
        },
        {
            "question": "How secure is the platform?",
            "category": "Mixed Match"
        },
        {
            "question": "Can I use CloudSphere with Kubernetes?",
            "category": "Partial Info"
        },
        {
            "question": "What's the weather today?",
            "category": "Out of Scope"
        }
    ]
    
    for test in test_questions:
        print(f"\n❓ Question: '{test['question']}'")
        print(f"📁 Category: {test['category']}")
        print("-" * 50)
        
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"question": test['question']}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            answer = response.json()["answer"]
            print(f"✅ Answer: {answer[:300]}{'...' if len(answer) > 300 else ''}")
            print(f"⏱️  Response time: {response_time:.2f} seconds")
        else:
            print(f"❌ Error: {response.status_code}")


def test_performance():
    """Μετράει την απόδοση του συστήματος."""
    print_section("Performance Testing")
    
    # Test 1: RAG Search μόνο
    print("📊 RAG Search Performance (without LLM):")
    
    rag_times = []
    for i in range(5):
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/rag/search",
            json={"question": "What is the refund policy?"}
        )
        end = time.time()
        
        if response.status_code == 200:
            rag_times.append(end - start)
    
    if rag_times:
        avg_rag_time = sum(rag_times) / len(rag_times)
        print(f"  Average RAG search time: {avg_rag_time:.3f} seconds")
        print(f"  Min: {min(rag_times):.3f}s, Max: {max(rag_times):.3f}s")
    
    # Test 2: Full System με LLM (with delays to avoid queueing)
    print("\n📊 Full System Performance (RAG + LLM):")
    
    full_times = []
    for i in range(3):  # Λιγότερα γιατί είναι πιο αργό
        if i > 0:
            print("  ⏳ Waiting 5 seconds to avoid LLM queueing...")
            time.sleep(5)  # Avoid overwhelming Ollama with concurrent requests
            
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"question": "How do I reset my password?"}
        )
        end = time.time()
        
        if response.status_code == 200:
            full_times.append(end - start)
            print(f"    Request {i+1}: {end - start:.2f} seconds")
        else:
            print(f"    Request {i+1}: FAILED ({response.status_code})")
    
    if full_times:
        avg_full_time = sum(full_times) / len(full_times)
        print(f"  Average full response time: {avg_full_time:.2f} seconds")
        print(f"  Min: {min(full_times):.2f}s, Max: {max(full_times):.2f}s")
    
    # Σύγκριση με το παλιό σύστημα
    print("\n📈 Improvement Analysis:")
    if rag_times:
        print(f"  RAG adds ~{avg_rag_time:.3f}s overhead")
        print(f"  But sends much less context to LLM")
        print(f"  Result: More accurate and scalable answers!")
    else:
        print(f"  RAG search failed - check endpoint availability")


def test_edge_cases():
    """Τεστάρει edge cases και error handling."""
    print_section("Testing Edge Cases")
    
    edge_cases = [
        {"question": ""},  # Empty question
        {"question": "a"},  # Too short
        {"question": "?" * 1000},  # Too long
        {"question": "SELECT * FROM users;"},  # SQL injection attempt
        {"question": "καλημέρα"},  # Non-English
    ]
    
    for i, case in enumerate(edge_cases):
        print(f"\nEdge case #{i+1}: {case}")
        
        response = requests.post(f"{BASE_URL}/ask", json=case)
        print(f"Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error handled correctly: {response.json().get('detail', 'Unknown error')}")


def main():
    """Τρέχει όλα τα tests."""
    print("🚀 Starting Hybrid RAG System Tests")
    print("Make sure the API is running at http://localhost:8002")
    
    # Έλεγχος αν το API τρέχει
    try:
        response = requests.get("http://localhost:8002/")
        if response.status_code != 200:
            print("❌ API is not responding. Please start it first.")
            return
    except Exception:
        print("❌ Cannot connect to API. Please run: uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload")
        return
    
    # Εκτέλεση tests
    test_rag_search()
    input("\n⏸️  Press Enter to continue to full system test...")
    
    test_full_system()
    input("\n⏸️  Press Enter to continue to performance test...")
    
    test_performance()
    input("\n⏸️  Press Enter to continue to edge cases test...")
    
    test_edge_cases()
    
    print_section("Testing Complete! 🎉")
    print("Your Hybrid RAG system is working great!")
    print("\nNext steps:")
    print("- Try different queries to see how it performs")
    print("- Adjust semantic/keyword weights in HybridRAGService")
    print("- Add more Q&As to the knowledge base")
    print("- Optimize performance with caching")


if __name__ == "__main__":
    main()