#!/usr/bin/env python3
"""
Simple performance test with delays to avoid LLM queueing
"""

import requests
import time

BASE_URL = "http://localhost:8002/api/v1"

def test_spaced_requests():
    """Test με διαστήματα για να μην κάνουμε queue το LLM"""
    print("🔍 Testing 3 spaced requests to avoid LLM queueing...\n")
    
    questions = [
        "What compliance certifications do you have?",
        "How do I reset my password?", 
        "What is the refund policy?"
    ]
    
    for i, question in enumerate(questions):
        print(f"📝 Request {i+1}: {question}")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                json={"question": question},
                timeout=120
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer_length = len(data.get('answer', ''))
                print(f"✅ Success in {duration:.1f}s (answer: {answer_length} chars)")
                print(f"📄 Answer: {data.get('answer', '')[:100]}...\n")
            else:
                print(f"❌ Failed: {response.status_code}\n")
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"⚠️  Error after {duration:.1f}s: {e}\n")
        
        # Wait between requests to avoid overloading Ollama
        if i < len(questions) - 1:
            print("⏳ Waiting 10 seconds before next request...")
            time.sleep(10)

if __name__ == "__main__":
    test_spaced_requests()
