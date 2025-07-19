#!/usr/bin/env python3
"""
Debug script to check what's stored in ChromaDB
"""
import sys
sys.path.append('.')
from app.chromadb_service import ChromaDBService

def main():
    # Ελέγχουμε τι έχει αποθηκευτεί
    chromadb = ChromaDBService()
    result = chromadb.collection.get()

    print(f"Total documents in ChromaDB: {len(result['documents'])}")
    print("\nFirst 5 stored documents:")
    for i, doc in enumerate(result['documents'][:5]):
        print(f"{i+1}. {doc[:80]}...")

    print("\nSearching for SOC 2 related content:")
    for i, doc in enumerate(result['documents']):
        if 'SOC' in doc or 'compliance' in doc or 'certification' in doc:
            print(f"Found at index {i}: {doc[:100]}...")

    print("\nSearching for refund related content:")
    for i, doc in enumerate(result['documents']):
        if 'refund' in doc.lower() or 'policy' in doc.lower():
            print(f"Found at index {i}: {doc[:100]}...")

if __name__ == "__main__":
    main()
