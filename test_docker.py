#!/usr/bin/env python3
"""
Test Docker deployment query - out of knowledge base question
"""

import requests
import json

def test_docker_query():
    """Test την Docker ερώτηση που είναι εκτός knowledge base"""
    
    url = "http://localhost:8002/api/v1/rag/search"
    query = "Can I deploy with Docker?"
    
    print(f"🔍 Testing query: '{query}'")
    print("=" * 60)
    
    response = requests.post(url, json={"question": query})
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"📊 Found {len(data['results'])} results\n")
        
        # Show top 5 results with detailed scores
        for i, result in enumerate(data['results'][:5]):
            print(f"#{i+1} [{result['match_type'].upper()}]")
            print(f"  Question: {result['question']}")
            print(f"  Answer preview: {result['answer'][:100]}...")
            print(f"  Scores:")
            print(f"    - Semantic: {result['scores']['semantic']:.4f}")
            print(f"    - Keyword: {result['scores']['keyword']:.4f}")
            print(f"    - Combined: {result['scores']['combined']:.4f}")
            print(f"  Explanation: {result['explanation']}")
            print("-" * 40)
            
        # Show search analysis
        exp = data['explanation']
        print(f"\n📈 Search Analysis:")
        print(f"  - Semantic matches: {exp['semantic_only']}")
        print(f"  - Keyword matches: {exp['keyword_only']}")
        print(f"  - Both methods: {exp['both_match']}")
        
        if exp['important_keywords']:
            print(f"  - Key terms found: {[term[0] for term in exp['important_keywords']]}")
        else:
            print(f"  - No significant keywords matched")
            
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_docker_query()
