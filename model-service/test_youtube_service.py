#!/usr/bin/env python3
"""
Test script for YouTube Analyzer Service

This script tests the YouTube analyzer service functionality.
"""

import requests
import json
import sys
from pathlib import Path

# Base URL for the model service
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed:")
            print(f"   Status: {data['status']}")
            print(f"   Models: {data['models']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_models():
    """Test the models endpoint"""
    print("\n🔍 Testing models endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/models")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Models info retrieved:")
            print(f"   Forecasting: {data['forecasting']['status']}")
            print(f"   Object Detection: {data['object_detection']['status']}")
            print(f"   YouTube Analyzer: {data['youtube_analyzer']['status']}")
            return True
        else:
            print(f"❌ Models info failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Models info error: {e}")
        return False

def test_youtube_analysis():
    """Test YouTube analysis endpoint"""
    print("\n🔍 Testing YouTube analysis endpoint...")
    
    # Test video (replace with a real YouTube URL that has subtitles)
    test_data = {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Replace with real URL
        "query": "What are the main topics discussed?",
        "max_chunks": 5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze/youtube",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ YouTube analysis successful:")
            print(f"   Video ID: {data['video_id']}")
            print(f"   Duration: {data.get('duration', 'Unknown')} seconds")
            print(f"   Chunks: {data['chunk_count']}")
            print(f"   Summary: {data['summary'][:100]}...")
            print(f"   Key points: {len(data['key_points'])} found")
            return True
        else:
            print(f"❌ YouTube analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ YouTube analysis error: {e}")
        return False

def test_invalid_url():
    """Test with invalid YouTube URL"""
    print("\n🔍 Testing invalid URL handling...")
    
    test_data = {
        "url": "https://invalid-url.com",
        "query": "Test query"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze/youtube",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 500:
            print("✅ Invalid URL properly handled with error response")
            return True
        else:
            print(f"⚠️ Unexpected response for invalid URL: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Invalid URL test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 YouTube Analyzer Service Tests")
    print("=" * 50)
    
    # Check if service is running
    print("Checking if service is running...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("❌ Service not running. Please start the service first:")
            print("   cd model-service")
            print("   python main.py")
            sys.exit(1)
    except:
        print("❌ Service not accessible. Please start the service first:")
        print("   cd model-service")
        print("   python main.py")
        sys.exit(1)
    
    print("✅ Service is running\n")
    
    # Run tests
    tests = [
        test_health,
        test_models,
        test_youtube_analysis,
        test_invalid_url
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check the service configuration.")

if __name__ == "__main__":
    main()