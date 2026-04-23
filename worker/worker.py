import os
import time
import json
import base64
import requests
import numpy as np
import cv2
import logging
from datetime import datetime

# Redis connection
import redis

# Import your webtest functions
from webtest import (
    CNICProcessor,
    detect_cnic_fields,
    process_cnic_front,
    detect_face_in_image,
    verify_face_live,
    extract_picture_from_cnic,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "your-secret-key")
REDIS_QUEUE = "cnic_tasks"

# Initialize Redis
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Initialize processor (loads YOLO + EasyOCR)
logger.info("Loading CNIC Processor (this may take a moment)...")
cnic_processor = CNICProcessor('runs/detect/train3/weights/best.pt')
logger.info("✅ CNIC Processor loaded successfully!")

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Convert base64 to OpenCV image"""
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    image_bytes = base64.b64decode(base64_str)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def process_extract_cnic(task_data: dict):
    """Process CNIC extraction task"""
    try:
        image = decode_base64_image(task_data["image_base64"])
        
        # Process CNIC
        detections = detect_cnic_fields(
            image, cnic_processor.model, cnic_processor.class_names
        )
        
        if not detections:
            return {"error": "No CNIC fields detected", "fields": {}}
        
        extracted_data, _, cnic_picture = process_cnic_front(image, cnic_processor)
        
        # Convert face to base64 if present
        cnic_face_base64 = None
        if cnic_picture is not None:
            _, buffer = cv2.imencode('.jpg', cnic_picture)
            cnic_face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "fields": extracted_data,
            "has_cnic_face": cnic_picture is not None,
            "cnic_face_base64": cnic_face_base64
        }
        
    except Exception as e:
        logger.error(f"Error in extract_cnic: {e}", exc_info=True)
        return {"error": str(e), "fields": {}}

def process_verify_face(task_data: dict):
    """Process face verification task"""
    try:
        cnic_image = decode_base64_image(task_data["cnic_base64"])
        selfie_image = decode_base64_image(task_data["selfie_base64"])
        
        # Extract face from CNIC
        detections = detect_cnic_fields(
            cnic_image, cnic_processor.model, cnic_processor.class_names
        )
        
        if not detections:
            return {"error": "No CNIC fields detected"}
        
        cnic_picture, _ = extract_picture_from_cnic(cnic_image, detections)
        if cnic_picture is None:
            return {"error": "Could not extract face from CNIC"}
        
        # Detect face in selfie
        detected_face_result = detect_face_in_image(selfie_image)
        if detected_face_result is None:
            return {"error": "No face detected in selfie image"}
        
        live_face, _ = detected_face_result
        
        # Verify faces
        verification = verify_face_live(cnic_picture, live_face)
        
        return verification
        
    except Exception as e:
        logger.error(f"Error in verify_face: {e}", exc_info=True)
        return {"error": str(e)}

def send_webhook(url: str, data: dict, api_key: str):
    """Send result to webhook"""
    try:
        data["api_key"] = api_key
        response = requests.post(url, json=data, timeout=30)
        if response.status_code != 200:
            logger.warning(f"Webhook failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Webhook error: {e}")

def main():
    logger.info(f"Worker started. Listening on Redis queue: {REDIS_QUEUE}")
    
    while True:
        try:
            # Pop task from Redis list (blocking pop with timeout)
            task_data_raw = redis_client.brpop(REDIS_QUEUE, timeout=5)
            
            if task_data_raw:
                _, task_json = task_data_raw
                task_data = json.loads(task_json)
                
                task_id = task_data["task_id"]
                task_type = task_data["type"]
                
                logger.info(f"Processing task {task_id} of type {task_type}")
                
                # Process based on type
                if task_type == "extract_cnic":
                    result = process_extract_cnic(task_data)
                    status = "completed" if "error" not in result else "failed"
                    
                    # Send result via webhook
                    if "webhook_url" in task_data:
                        send_webhook(task_data["webhook_url"], result, WORKER_API_KEY)
                    else:
                        # Fallback: store directly in Redis
                        redis_client.setex(f"task:{task_id}", 3600, json.dumps({
                            "task_id": task_id,
                            "status": status,
                            "result": result if status == "completed" else None,
                            "error": result.get("error") if status == "failed" else None,
                            "completed_at": datetime.now().isoformat()
                        }))
                
                elif task_type == "verify_face":
                    result = process_verify_face(task_data)
                    status = "completed" if "error" not in result else "failed"
                    
                    if "webhook_url" in task_data:
                        send_webhook(task_data["webhook_url"], result, WORKER_API_KEY)
                
                logger.info(f"Task {task_id} completed")
            
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            time.sleep(1)

if __name__ == "__main__":
    main()