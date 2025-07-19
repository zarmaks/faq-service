#!/usr/bin/env python3
"""
Single request test to check LLM performance without concurrent load
"""

import requests
import json
import time

def test_single_request():
    url = "http://localhost:8002/api/v1/ask"
    
    payload = {
        "question": "What compliance certifications do you have?"
    }
    
    print("🔍 Testing single request...")
    print(f"Question: {payload['question']}")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Response time: {duration:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Answer length: {len(data.get('answer', ''))}")
            print(f"📝 Answer: {data.get('answer', 'No answer')}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"⚠️  Exception after {duration:.2f}s: {e}")

if __name__ == "__main__":
    test_single_request()
