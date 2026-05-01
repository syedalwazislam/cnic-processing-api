import requests
import time
import json

API_URL = "https://cnic-processing-api-production.up.railway.app"

def test_connection():
    """Test API and Redis connection"""
    print("="*50)
    print("Testing API Connection")
    print("="*50)
    
    # Check API health
    response = requests.get(f"{API_URL}/health")
    health = response.json()
    print(f"✅ API Status: {health['status']}")
    print(f"✅ Redis Status: {health['redis']}")
    print(f"✅ Service: {health['service']}")
    
    return health['redis'] == 'connected'

def submit_and_check(image_path):
    """Submit CNIC and check result"""
    
    print("\n" + "="*50)
    print("Submitting CNIC for Processing")
    print("="*50)
    
    # Read image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    print(f"✅ Loaded image: {len(image_data)} bytes")
    
    # Submit to API
    print("\n📤 Submitting to Railway API...")
    files = {'cnic_image': ('cnic.jpg', image_data, 'image/jpeg')}
    response = requests.post(f"{API_URL}/extract-cnic", files=files)
    
    if response.status_code != 200:
        print(f"❌ Failed: {response.text}")
        return
    
    task = response.json()
    task_id = task['task_id']
    print(f"✅ Task ID: {task_id}")
    print(f"   Status: {task['status']}")
    
    # Poll for results
    print("\n⏳ Waiting for EC2 worker to process...")
    print("   (Check EC2 logs: sudo docker logs cnic-worker -f)")
    
    for i in range(45):  # 90 seconds max
        time.sleep(2)
        
        result_response = requests.get(f"{API_URL}/result/{task_id}")
        
        if result_response.status_code == 200:
            result = result_response.json()
            status = result.get('status')
            
            print(f"   [{i+1}] Status: {status}")
            
            if status == 'completed':
                print("\n" + "="*50)
                print("✅ SUCCESS! CNIC Data Extracted")
                print("="*50)
                
                fields = result.get('result', {}).get('fields', {})
                if fields:
                    print("\n📋 Extracted Information:")
                    for key, value in fields.items():
                        print(f"   {key}: {value}")
                else:
                    print("⚠️ No fields extracted")
                
                return True
                
            elif status == 'failed':
                print(f"\n❌ Failed: {result.get('error')}")
                return False
        
    print("\n❌ Timeout - Check EC2 worker logs")
    return False

if __name__ == "__main__":
    # Test connection
    if not test_connection():
        print("❌ API not available")
        exit(1)
    
    # Test with CNIC image
    image_path = input("\n📷 Enter path to CNIC image: ").strip()
    if image_path:
        submit_and_check(image_path)
    else:
        print("No image provided")