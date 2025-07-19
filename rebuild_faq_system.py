#!/usr/bin/env python3
"""
Rebuild FAQ Service από την αρχή με τη διορθωμένη similarity calculation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import time
from app.rag_service import HybridRAGService


def rebuild_faq_system():
    """Ξαναφτιάχνει ολόκληρο το FAQ system από την αρχή."""
    print("🚀 Rebuilding FAQ Service from Scratch")
    print("=" * 60)
    
    print("📂 Step 1: Checking knowledge base...")
    kb_path = "data/knowledge_base.txt"
    if not os.path.exists(kb_path):
        print("❌ Knowledge base file not found!")
        return False
    
    # Check file size
    file_size = os.path.getsize(kb_path)
    print(f"✅ Knowledge base found: {file_size} bytes")
    
    print("\n🧠 Step 2: Initializing RAG Service...")
    print("This will:")
    print("  - Parse Q&A pairs from knowledge base")
    print("  - Create embeddings with Ollama") 
    print("  - Store in ChromaDB with NEW similarity calculation")
    print("  - Build TF-IDF index")
    
    start_time = time.time()
    
    try:
        # Initialize RAG service (this rebuilds everything)
        rag_service = HybridRAGService(
            knowledge_base_path=kb_path,
            semantic_weight=0.6,  # 60% semantic, 40% keyword
            keyword_weight=0.4
        )
        
        build_time = time.time() - start_time
        print(f"\n✅ RAG Service initialized in {build_time:.1f} seconds")
        
        # Get stats
        print("\n📊 Step 3: System Statistics")
        parser_stats = rag_service.parser.get_stats()
        chromadb_stats = rag_service.chromadb_service.get_stats()
        
        print(f"  📖 Knowledge Base:")
        print(f"    - Total Q&A pairs: {parser_stats['total_pairs']}")
        print(f"    - Total characters: {parser_stats['total_characters']:,}")
        print(f"    - Avg question length: {parser_stats['average_question_length']}")
        print(f"    - Avg answer length: {parser_stats['average_answer_length']}")
        
        print(f"  🗄️  ChromaDB:")
        print(f"    - Total embeddings: {chromadb_stats['total_embeddings']}")
        print(f"    - Collection name: {chromadb_stats['collection_name']}")
        
        return rag_service
        
    except Exception as e:
        print(f"❌ Error initializing RAG service: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_new_similarities(rag_service):
    """Δοκιμάζει τις νέες similarity calculations."""
    print("\n🧪 Step 4: Testing New Similarity Calculations")
    print("=" * 60)
    
    # Test queries - mix of in-knowledge and out-of-knowledge
    test_cases = [
        {
            "query": "SOC 2 compliance certification",
            "expected": "high",
            "description": "Should find exact match in KB"
        },
        {
            "query": "What is SOC2 compliance?", 
            "expected": "high",
            "description": "Slightly different wording, should still match"
        },
        {
            "query": "password reset help",
            "expected": "medium", 
            "description": "Related to account recovery in KB"
        },
        {
            "query": "Can I deploy with Docker?",
            "expected": "low",
            "description": "Out of knowledge base - deployment topic"
        },
        {
            "query": "What programming languages do you support?",
            "expected": "low", 
            "description": "Out of knowledge base - technical specs"
        }
    ]
    
    print(f"{'Query':<40} {'Top Score':<10} {'Match Type':<12} {'Status'}")
    print("-" * 75)
    
    results_summary = []
    
    for test_case in test_cases:
        query = test_case["query"]
        expected = test_case["expected"]
        
        # Search with RAG service
        search_results = rag_service.search(query, n_results=3)
        
        if search_results:
            top_result = search_results[0]
            score = top_result.combined_score
            match_type = top_result.match_type
            
            # Determine if result matches expectation
            if expected == "high" and score > 0.3:
                status = "✅ GOOD"
            elif expected == "medium" and 0.1 < score < 0.3:
                status = "✅ GOOD"  
            elif expected == "low" and score < 0.1:
                status = "✅ GOOD"
            else:
                status = "⚠️  CHECK"
                
        else:
            score = 0.0
            match_type = "none"
            status = "❌ NO_RESULTS"
        
        print(f"{query:<40} {score:<10.3f} {match_type:<12} {status}")
        
        results_summary.append({
            "query": query,
            "score": score,
            "expected": expected,
            "status": status
        })
    
    # Summary
    print(f"\n📈 Results Analysis:")
    good_results = [r for r in results_summary if r["status"] == "✅ GOOD"]
    print(f"  ✅ Good results: {len(good_results)}/{len(results_summary)}")
    
    if len(good_results) == len(results_summary):
        print("  🎉 All similarity calculations working correctly!")
    else:
        print("  ⚠️  Some results need review")
    
    return results_summary


def quick_search_demo(rag_service):
    """Γρήγορο demo των search capabilities."""
    print("\n🔍 Step 5: Quick Search Demo")
    print("=" * 60)
    
    demo_queries = [
        "What is the refund policy?",
        "forgotten password",
        "Can I deploy with Docker?"  # Out of KB
    ]
    
    for query in demo_queries:
        print(f"\n💬 Query: '{query}'")
        results = rag_service.search(query, n_results=2)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.combined_score:.3f} | Type: {result.match_type}")
            print(f"     Q: {result.qa_pair.question[:60]}...")
            print(f"     A: {result.qa_pair.answer[:80]}...")


if __name__ == "__main__":
    print("Starting FAQ Service Rebuild...")
    
    # Step 1-3: Rebuild system
    rag_service = rebuild_faq_system()
    if not rag_service:
        print("❌ Failed to rebuild system. Exiting.")
        sys.exit(1)
    
    # Step 4: Test similarities  
    test_results = test_new_similarities(rag_service)
    
    # Step 5: Demo
    quick_search_demo(rag_service)
    
    print("\n🎯 Final Status:")
    print("✅ FAQ Service rebuilt successfully")
    print("✅ New similarity calculation active")
    print("✅ Ready for testing with improved scores!")
    print("\nNext steps:")
    print("- Run full tests with: python app/test_rag.py")
    print("- Start API server with: python -m app.main")
