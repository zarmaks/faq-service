#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from app.rag_service import HybridRAGService

def main():
    print("🔧 Testing SOC 2 compliance query...")
    
    # Initialize RAG service
    rag = HybridRAGService("data/knowledge_base.txt")
    
    # Test SOC 2 query
    query = "SOC 2 compliance certification"
    print(f"🔍 Testing: '{query}'")
    print("-" * 50)
    
    results = rag.search(query, n_results=5)
    
    if results:
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  #{i} [{result.match_type}]")
            print(f"  Q: {result.qa_pair.question[:60]}...")
            print(f"  Semantic: {result.semantic_score:.3f}")
            print(f"  Keyword: {result.keyword_score:.3f}")
            print(f"  Combined: {result.combined_score:.3f}")
            print(f"  Explanation: {result.explanation}")
            print()
    else:
        print("❌ No results found!")

if __name__ == "__main__":
    main()
