import requests
import base64
import time
import json
import sys

# ===== CONFIGURATION =====
# Replace with your actual Railway API URL
API_URL = "https://cnic-processing-api-production.up.railway.app"  # ← CHANGE THIS

def test_health():
    """Test if API is reachable"""
    print("🔍 Checking API health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"✅ API is running: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Cannot reach API: {e}")
        return False

def test_extract_cnic(image_path):
    """Test CNIC extraction end-to-end"""
    
    print("\n" + "="*50)
    print("📸 Testing CNIC Extraction")
    print("="*50)
    
    # Check if image exists
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        print(f"✅ Loaded image: {image_path} ({len(image_data)} bytes)")
    except Exception as e:
        print(f"❌ Cannot read image: {e}")
        return None
    
    # Step 1: Submit task to API
    print("\n📤 Submitting task to API...")
    files = {'cnic_image': ('cnic.jpg', image_data, 'image/jpeg')}
    
    try:
        response = requests.post(f"{API_URL}/extract-cnic", files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ API error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
        
        task = response.json()
        task_id = task.get('task_id')
        print(f"✅ Task submitted!")
        print(f"   Task ID: {task_id}")
        print(f"   Status: {task.get('status')}")
        
    except Exception as e:
        print(f"❌ Failed to submit: {e}")
        return None
    
    # Step 2: Poll for results
    print("\n⏳ Waiting for worker to process...")
    print("   (Check Hugging Face logs for progress)")
    
    max_attempts = 30  # 60 seconds max
    for i in range(max_attempts):
        time.sleep(2)
        
        try:
            result_response = requests.get(f"{API_URL}/result/{task_id}", timeout=5)
            
            if result_response.status_code == 200:
                result = result_response.json()
                status = result.get('status')
                
                print(f"   Attempt {i+1}: Status = {status}")
                
                if status == 'completed':
                    print("\n" + "="*50)
                    print("✅ TASK COMPLETED SUCCESSFULLY!")
                    print("="*50)
                    
                    # Print extracted data
                    result_data = result.get('result', {})
                    fields = result_data.get('fields', {})
                    
                    if fields:
                        print("\n📋 EXTRACTED CNIC DATA:")
                        print("-"*30)
                        for key, value in fields.items():
                            print(f"   {key}: {value}")
                    else:
                        print("\n⚠️ No fields extracted")
                        print(f"   Raw result: {json.dumps(result_data, indent=2)}")
                    
                    return result
                    
                elif status == 'failed':
                    print(f"\n❌ Task failed: {result.get('error')}")
                    return None
            else:
                print(f"   Attempt {i+1}: Result not ready (HTTP {result_response.status_code})")
                
        except Exception as e:
            print(f"   Attempt {i+1}: Error checking result: {e}")
    
    print("\n❌ Timeout waiting for task to complete")
    return None

def test_face_verification(cnic_path, selfie_path):
    """Test face verification"""
    
    print("\n" + "="*50)
    print("👤 Testing Face Verification")
    print("="*50)
    
    # Check images
    try:
        with open(cnic_path, 'rb') as f:
            cnic_data = f.read()
        with open(selfie_path, 'rb') as f:
            selfie_data = f.read()
        print(f"✅ Loaded CNIC image ({len(cnic_data)} bytes)")
        print(f"✅ Loaded selfie ({len(selfie_data)} bytes)")
    except Exception as e:
        print(f"❌ Cannot read images: {e}")
        return None
    
    # Submit task
    print("\n📤 Submitting verification task...")
    files = {
        'cnic_image': ('cnic.jpg', cnic_data, 'image/jpeg'),
        'selfie_image': ('selfie.jpg', selfie_data, 'image/jpeg')
    }
    
    try:
        response = requests.post(f"{API_URL}/verify-face", files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ API error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
        
        task = response.json()
        task_id = task.get('task_id')
        print(f"✅ Task submitted! Task ID: {task_id}")
        
    except Exception as e:
        print(f"❌ Failed to submit: {e}")
        return None
    
    # Poll for results
    print("\n⏳ Waiting for verification...")
    
    for i in range(30):
        time.sleep(2)
        
        try:
            result_response = requests.get(f"{API_URL}/result/{task_id}", timeout=5)
            
            if result_response.status_code == 200:
                result = result_response.json()
                status = result.get('status')
                
                if status == 'completed':
                    print("\n✅ VERIFICATION COMPLETED!")
                    verification = result.get('result', {})
                    print(f"   Match: {verification.get('final_verification')}")
                    print(f"   Confidence: {verification.get('confidence', 'N/A')}%")
                    return result
                elif status == 'failed':
                    print(f"\n❌ Verification failed: {result.get('error')}")
                    return None
                    
        except Exception as e:
            print(f"   Attempt {i+1}: Error: {e}")
    
    print("\n❌ Timeout waiting for verification")
    return None

def check_redis_queue():
    """Check if there are tasks in queue (optional)"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"\n📊 API Status: Redis={health.get('redis')}")
    except:
        pass

if __name__ == "__main__":
    print("="*50)
    print("🚀 CNIC SYSTEM TEST")
    print("="*50)
    print(f"API URL: {API_URL}")
    
    # Check API health
    if not test_health():
        print("\n❌ Cannot proceed - API not reachable")
        print("   Make sure your Railway API is deployed and running")
        sys.exit(1)
    
    # Ask for test image
    print("\n" + "="*50)
    cnic_image = input("📷 Enter path to CNIC image file: ").strip()
    
    if cnic_image:
        test_extract_cnic(cnic_image)
    else:
        print("⏭️ Skipping CNIC extraction test")
    
    # Ask for face verification test
    print("\n" + "="*50)
    verify = input("👤 Test face verification? (y/n): ").strip().lower()
    
    if verify == 'y':
        cnic_img = input("CNIC image path: ").strip()
        selfie_img = input("Selfie image path: ").strip()
        if cnic_img and selfie_img:
            test_face_verification(cnic_img, selfie_img)
    
    print("\n✨ Test complete!")