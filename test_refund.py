#!/usr/bin/env python3
"""
Test script για in-scope query - refund policy
"""
import requests
import json

def test_refund_query():
    url = "http://localhost:8002/api/v1/rag/search"
    query = "What is the refund policy?"
    
    print(f"🔍 Testing query: '{query}'")
    print("=" * 60)
    
    try:
        response = requests.post(url, json={"question": query})
        response.raise_for_status()
        
        result = response.json()
        print(f"📊 Found {len(result['results'])} results")
        
        for i, result_item in enumerate(result['results'][:5]):
            print(f"#{i+1} [{result_item['match_type'].upper()}]")
            print(f"  Question: {result_item['question']}")
            print(f"  Answer preview: {result_item['answer'][:80]}...")
            print(f"  Scores:")
            print(f"    - Semantic: {result_item['semantic_score']:.4f}")
            print(f"    - Keyword: {result_item['keyword_score']:.4f}")
            print(f"    - Combined: {result_item['combined_score']:.4f}")
            
            # Explain match type
            semantic = result_item['semantic_score'] 
            keyword = result_item['keyword_score']
            if semantic > 0.01 and keyword > 0.01:
                print(f"  Explanation: Both semantic ({semantic:.2f}) and keyword ({keyword:.2f}) match")
            elif semantic > 0.01:
                print(f"  Explanation: Strong semantic match (score: {semantic:.2f})")
            elif keyword > 0.01:
                print(f"  Explanation: Strong keyword match (score: {keyword:.2f})")
            else:
                print(f"  Explanation: Weak match")
            print("-" * 40)
        
        # Analysis
        semantic_matches = sum(1 for r in result['results'] if r['semantic_score'] > 0.01)
        keyword_matches = sum(1 for r in result['results'] if r['keyword_score'] > 0.01) 
        both_matches = sum(1 for r in result['results'] if r['semantic_score'] > 0.01 and r['keyword_score'] > 0.01)
        
        print("📈 Search Analysis:")
        print(f"  - Semantic matches: {semantic_matches}")
        print(f"  - Keyword matches: {keyword_matches}")
        print(f"  - Both methods: {both_matches}")
        
        # Show key terms
        key_terms = []
        for r in result['results']:
            if r['keyword_score'] > 0.01:
                # This is a simplistic way to guess which terms matched
                if 'refund' in r['question'].lower() or 'refund' in r['answer'].lower():
                    key_terms.append('refund')
                if 'policy' in r['question'].lower() or 'policy' in r['answer'].lower():
                    key_terms.append('policy')
        key_terms = list(set(key_terms))
        if key_terms:
            print(f"  - Key terms found: {key_terms}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_refund_query()
