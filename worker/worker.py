import os
import time
import json
import base64
import requests
import numpy as np
import cv2
import logging
import re
import io
from datetime import datetime

# Redis connection
import redis

# Gemini — for CNIC back-side address extraction (free, 1500 req/day)
# Get your free key at: https://aistudio.google.com/app/apikey
# Set on EC2: export GEMINI_API_KEY="AIza..."
try:
    import google.generativeai as genai
    from PIL import Image as PILImage
    _GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
    if _GEMINI_KEY:
        genai.configure(api_key=_GEMINI_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash-lite")
        GEMINI_AVAILABLE = True
        logging.getLogger(__name__).info("✅ Gemini 2.5 Flash Lite ready for back-side OCR")
    else:
        GEMINI_AVAILABLE = False
        logging.getLogger(__name__).warning(
            "⚠️  GEMINI_API_KEY not set — extract_back tasks will fail. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
except ImportError:
    GEMINI_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "⚠️  google-generativeai not installed. "
        "Run: pip install google-generativeai pillow"
    )

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
    task_id = task_data["task_id"]
    
    try:
        image = decode_base64_image(task_data["image_base64"])
        
        # Process CNIC
        detections = detect_cnic_fields(
            image, cnic_processor.model, cnic_processor.class_names
        )
        
        if not detections:
            error_msg = "No CNIC fields detected in image"
            # Store error in Redis
            task_error = {
                "task_id": task_id,
                "status": "failed",
                "error": error_msg,
                "completed_at": datetime.now().isoformat()
            }
            redis_client.setex(f"task:{task_id}", 3600, json.dumps(task_error))
            return {"error": error_msg}
        
        extracted_data, _, cnic_picture = process_cnic_front(image, cnic_processor)
        
        # Prepare result
        result = {
            "fields": extracted_data,
            "has_cnic_face": cnic_picture is not None
        }
        
        # STORE IN REDIS - THIS IS THE KEY FIX
        task_result = {
            "task_id": task_id,
            "status": "completed",
            "result": result,
            "completed_at": datetime.now().isoformat()
        }
        redis_client.setex(f"task:{task_id}", 3600, json.dumps(task_result))
        logger.info(f"✅ Task {task_id} result stored in Redis")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in extract_cnic: {e}", exc_info=True)
        error_msg = str(e)
        
        # Store error in Redis
        task_error = {
            "task_id": task_id,
            "status": "failed",
            "error": error_msg,
            "completed_at": datetime.now().isoformat()
        }
        redis_client.setex(f"task:{task_id}", 3600, json.dumps(task_error))
        
        return {"error": error_msg}

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

_BACK_PROMPT = """
You are an OCR expert for Pakistani CNIC (National Identity Card) documents.

This image is the BACK side of a Pakistani CNIC card.

The card contains TWO address blocks in Urdu (Nastaliq script), read right-to-left:
  1. موجودہ پتہ  (Mojooda Pata)  = CURRENT ADDRESS   — upper section
  2. مستقل پتہ  (Mustaqil Pata) = PERMANENT ADDRESS  — lower section

A horizontal line separates the two address blocks.
The CNIC number (format XXXXX-XXXXXXX-X) appears at the top right.
A barcode serial number appears at the bottom right.

Instructions:
- Read all Urdu text carefully (right-to-left)
- Preserve full Urdu text exactly as printed
- Also provide a romanized/English transliteration for each address
- Use null for any field that is not visible or unclear

Return ONLY a valid JSON object with NO markdown fences or extra text:

{
  "cnic_number": "XXXXX-XXXXXXX-X or null",
  "mojooda_pata_urdu": "current address in Urdu script",
  "mojooda_pata_roman": "current address romanized in English",
  "mustaqil_pata_urdu": "permanent address in Urdu script",
  "mustaqil_pata_roman": "permanent address romanized in English",
  "barcode_number": "number near barcode or null",
  "confidence": "high or medium or low"
}
"""

def _parse_gemini_json(raw: str) -> dict:
    """Safely extract JSON from Gemini response, stripping any markdown fences."""
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"raw_text": raw, "parse_error": "Could not parse JSON", "confidence": "low"}


def process_extract_cnic_back(task_data: dict):
    """
    Process CNIC back-side image to extract موجودہ پتہ and مستقل پتہ using Gemini.
    task_data keys expected:
        task_id   - str
        task_type - "extract_back"
        image     - base64-encoded image bytes (str)   ← set by api.py
    """
    task_id = task_data["task_id"]

    if not GEMINI_AVAILABLE:
        error_result = {
            "task_id": task_id,
            "status": "failed",
            "error": (
                "Gemini is not configured on this worker. "
                "Install google-generativeai and set GEMINI_API_KEY."
            ),
            "completed_at": datetime.now().isoformat(),
        }
        redis_client.setex(f"task:{task_id}", 3600, json.dumps(error_result))
        return {"error": error_result["error"]}

    try:
        # Decode image (api.py sends it as plain base64 under key "image")
        raw_b64 = task_data.get("image") or task_data.get("image_base64", "")
        if "," in raw_b64:                          # strip data-URI prefix if present
            raw_b64 = raw_b64.split(",")[1]
        image_bytes = base64.b64decode(raw_b64)
        pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.info(f"[BACK] Task {task_id} — image size: {pil_image.size}")

        # Call Gemini 1.5 Flash
        response = GEMINI_MODEL.generate_content(
            [_BACK_PROMPT, pil_image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        raw_text = response.text.strip()
        logger.info(f"[BACK] Gemini raw response: {raw_text}")

        fields = _parse_gemini_json(raw_text)

        result = {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "task_type": "extract_back",
                "fields": fields,
            },
            "completed_at": datetime.now().isoformat(),
        }
        redis_client.setex(f"task:{task_id}", 3600, json.dumps(result))
        logger.info(f"✅ [BACK] Task {task_id} stored in Redis")
        return result

    except Exception as e:
        logger.error(f"[BACK] Error in extract_cnic_back: {e}", exc_info=True)
        error_result = {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        }
        redis_client.setex(f"task:{task_id}", 3600, json.dumps(error_result))
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
                task_type = task_data.get("type") or task_data.get("task_type", "extract_cnic")

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

                elif task_type == "extract_back":
                    # CNIC back-side: extract موجودہ پتہ + مستقل پتہ via Gemini
                    process_extract_cnic_back(task_data)   # stores result in Redis internally
                
                logger.info(f"Task {task_id} completed")
            
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            time.sleep(1)

if __name__ == "__main__":
    main()