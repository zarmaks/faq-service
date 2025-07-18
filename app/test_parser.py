"""
Test script για τον Knowledge Base Parser.

Τρέξε το με: python test_parser.py
"""

from app.kb_parser import KnowledgeBaseParser, load_knowledge_base
import json

def test_parser():
    """Τεστάρει τον parser με το πραγματικό knowledge base."""
    
    print("🔍 Testing Knowledge Base Parser\n")
    
    # Δημιουργούμε τον parser
    parser = KnowledgeBaseParser("data/knowledge_base.txt")
    
    # Parse το αρχείο
    qa_pairs = parser.parse()
    
    print(f"✅ Βρέθηκαν {len(qa_pairs)} Q&A pairs\n")
    
    # Δείχνουμε τα πρώτα 3 για επιβεβαίωση
    print("📋 Πρώτα 3 Q&A pairs:")
    print("-" * 50)
    
    for i, qa in enumerate(qa_pairs[:3]):
        print(f"\n🆔 ID: {qa.id}")
        print(f"❓ Q: {qa.question[:100]}...")
        print(f"💡 A: {qa.answer[:100]}...")
    
    # Στατιστικά
    print("\n📊 Στατιστικά Knowledge Base:")
    print("-" * 50)
    stats = parser.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test αναζήτησης με keyword
    print("\n🔎 Test keyword search για 'refund':")
    print("-" * 50)
    results = parser.search_by_keyword("refund")
    print(f"Βρέθηκαν {len(results)} αποτελέσματα")
    for qa in results:
        print(f"- {qa.question[:60]}...")
    
    # Test to_text() method
    print("\n📝 Παράδειγμα πώς φαίνεται ένα chunk:")
    print("-" * 50)
    print(qa_pairs[0].to_text())
    
    print("\n✅ Parser testing completed successfully!")

if __name__ == "__main__":
    test_parser()