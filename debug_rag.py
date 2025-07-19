"""
Debug script για το RAG service.
"""

import os
import sys
sys.path.append(".")

from app.rag_service import HybridRAGService

def debug_rag():
    """Δοκιμάζει το RAG service απευθείας."""
    
    knowledge_base_path = os.path.join("data", "knowledge_base.txt")
    
    print("🔧 Debugging RAG Service...")
    print(f"Knowledge base path: {knowledge_base_path}")
    
    try:
        # Initialize RAG service
        rag = HybridRAGService(knowledge_base_path)
        
        # Test queries
        test_queries = [
            "What is the refund policy?",
            "How much does Professional tier cost?",
            "Do you offer a free trial?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            print("-" * 50)
            
            # Search without threshold - let LLM decide
            results = rag.search(query, n_results=5)
            
            print(f"Found {len(results)} results:")
            
            for i, result in enumerate(results):
                print(f"\n  #{i+1} [{result.match_type.upper()}]")
                print(f"  Q: {result.qa_pair.question[:60]}...")
                print(f"  Semantic: {result.semantic_score:.3f}")
                print(f"  Keyword: {result.keyword_score:.3f}")
                print(f"  Combined: {result.combined_score:.3f}")
                print(f"  Explanation: {result.explanation}")
            
            if not results:
                print("  No results found - investigating...")
                
                # Test semantic search directly
                print("\n  🧪 Testing semantic search directly...")
                embedding = rag.embeddings_service.create_embedding(query)
                semantic_results = rag.chromadb_service.search(embedding, n_results=3)
                print(f"  Semantic search returned {len(semantic_results)} results")
                
                for j, sr in enumerate(semantic_results):
                    print(f"    #{j+1}: similarity={sr.similarity:.3f}, qa_id={sr.qa_id}")
                
                # Test keyword search directly
                print("\n  🧪 Testing keyword search directly...")
                keyword_results = rag.tfidf_service.search(query, n_results=3)
                print(f"  Keyword search returned {len(keyword_results)} results")
                
                for j, (qa_id, score) in enumerate(keyword_results):
                    print(f"    #{j+1}: score={score:.3f}, qa_id={qa_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_rag()
