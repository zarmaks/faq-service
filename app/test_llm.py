"""
Test script για το LLM integration.

Τρέξε το με: python test_llm.py
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8001"

def test_llm_connection():
    """Τεστάρει αν το LLM είναι συνδεδεμένο."""
    print("🔍 Testing LLM connection...")
    
    response = requests.get(f"{BASE_URL}/api/v1/llm/test")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ LLM Connection Status:", data.get("status"))
        print("📦 Model:", data.get("model"))
        print("🖥️  Ollama URL:", data.get("ollama_url"))
        print("📋 Available Models:", data.get("available_models"))
        return True
    else:
        print("❌ LLM connection failed:", response.text)
        return False

def test_faq_questions():
    """Δοκιμάζει διάφορες ερωτήσεις στο FAQ system."""
    
    # Λίστα με test ερωτήσεις
    test_questions = [
        "What is CloudSphere Platform?",
        "How much does the Professional tier cost?",
        "Do you offer a free trial?",
        "What payment methods do you accept?",
        "How can I reset my password?",
        "What is your refund policy?",
        "Do you support MFA?",
        "What are the API rate limits?",
        "Can I use CloudSphere with Docker?",  # Ερώτηση εκτός knowledge base
        "Hello, how are you?"  # Άσχετη ερώτηση
    ]
    
    print("\n📝 Testing FAQ questions...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Question {i}/{len(test_questions)} ---")
        print(f"❓ Question: {question}")
        
        # Μετράμε τον χρόνο απόκρισης
        start_time = time.time()
        
        # Στέλνουμε την ερώτηση
        response = requests.post(
            f"{BASE_URL}/api/v1/ask",
            json={"question": question}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            answer = response.json()["answer"]
            print(f"✅ Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"⏱️  Response time: {response_time:.2f} seconds")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")

def test_history():
    """Τεστάρει το history endpoint."""
    print("\n📚 Testing history endpoint...")
    
    response = requests.get(f"{BASE_URL}/api/v1/history?n=5")
    
    if response.status_code == 200:
        history = response.json()
        print(f"✅ Found {len(history)} questions in history:")
        
        for item in history:
            print(f"\n🕐 {item['timestamp']}")
            print(f"   Q: {item['question_text'][:100]}...")
            print(f"   A: {item['answer_text'][:100]}...")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def main():
    """Τρέχει όλα τα tests."""
    print("🚀 Starting LLM Integration Tests\n")
    
    # Test 1: Έλεγχος σύνδεσης
    if not test_llm_connection():
        print("\n⚠️  Cannot connect to LLM. Make sure Ollama is running!")
        print("Run: ollama serve")
        return
    
    # Test 2: FAQ ερωτήσεις
    test_faq_questions()
    
    # Test 3: History
    test_history()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()