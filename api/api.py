import logging
import asyncio
import os
import base64
import requests
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import redis
import json
import uuid

from fastapi.middleware.cors import CORSMiddleware

# Add this right after `app = FastAPI(...)` line:


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "your-secret-key")  # For worker->API auth

# Connect to Redis (task queue)
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info(f"✅ Connected to Redis at {REDIS_URL}")
except Exception as e:
    logger.error(f"❌ Failed to connect to Redis: {e}")
    redis_client = None

# ---------------------------------------------------------------------------
# App & Models
# ---------------------------------------------------------------------------
app = FastAPI(title="CNIC Processing API (Lightweight)", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods
    allow_headers=["*"],       # Allow all headers
)

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskResultResponse(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')

async def read_upload_as_base64(upload: UploadFile) -> str:
    """Read uploaded file and convert to base64"""
    contents = await upload.read()
    return image_to_base64(contents)

def store_task_metadata(task_id: str, status: str = "pending"):
    """Store task metadata in Redis"""
    if redis_client:
        task_data = {
            "task_id": task_id,
            "status": status,
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None
        }
        redis_client.setex(f"task:{task_id}", 3600, json.dumps(task_data))  # 1 hour expiry

def update_task_result(task_id: str, result: dict, status: str = "completed"):
    """Update task result in Redis"""
    if redis_client:
        task_data = redis_client.get(f"task:{task_id}")
        if task_data:
            data = json.loads(task_data)
            data["status"] = status
            data["result"] = result
            data["completed_at"] = datetime.now().isoformat()
            redis_client.setex(f"task:{task_id}", 3600, json.dumps(data))

def update_task_error(task_id: str, error: str):
    """Update task error in Redis"""
    if redis_client:
        task_data = redis_client.get(f"task:{task_id}")
        if task_data:
            data = json.loads(task_data)
            data["status"] = "failed"
            data["error"] = error
            data["completed_at"] = datetime.now().isoformat()
            redis_client.setex(f"task:{task_id}", 3600, json.dumps(data))

# ---------------------------------------------------------------------------
# Webhook endpoint for worker to report results
# ---------------------------------------------------------------------------
@app.post("/webhook/result/{task_id}")
async def worker_result_webhook(
    task_id: str, 
    result: dict,
    api_key: str
):
    """Worker calls this webhook when task is complete"""
    if api_key != WORKER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    update_task_result(task_id, result)
    logger.info(f"Webhook received for task {task_id}")
    return {"status": "ok"}

@app.post("/webhook/error/{task_id}")
async def worker_error_webhook(
    task_id: str,
    error: dict,
    api_key: str
):
    """Worker calls this webhook when task fails"""
    if api_key != WORKER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    update_task_error(task_id, error.get("message", "Unknown error"))
    logger.info(f"Error webhook received for task {task_id}: {error}")
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "redis": "connected" if redis_client else "disconnected",
        "service": "api-gateway"
    }

@app.post("/extract-cnic", response_model=TaskResponse)
async def extract_cnic(
    background_tasks: BackgroundTasks,
    cnic_image: UploadFile = File(...),
):
    """
    Submit CNIC for extraction (async processing)
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Read image and convert to base64
    image_base64 = await read_upload_as_base64(cnic_image)
    
    # Store task metadata
    store_task_metadata(task_id)
    
    # Add to Redis queue for worker to pick up
    task_data = {
        "task_id": task_id,
        "type": "extract_cnic",
        "image_base64": image_base64,
        "webhook_url": f"https://your-railway-app.railway.app/webhook/result/{task_id}",
        "webhook_error_url": f"https://your-railway-app.railway.app/webhook/error/{task_id}"
    }
    
    # Push to Redis list (queue)
    redis_client.lpush("cnic_tasks", json.dumps(task_data))
    
    logger.info(f"Task {task_id} queued for CNIC extraction")
    
    return TaskResponse(
        task_id=task_id,
        status="queued",
        message="CNIC extraction task has been queued"
    )

@app.post("/verify-face", response_model=TaskResponse)
async def verify_face(
    background_tasks: BackgroundTasks,
    cnic_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...),
):
    """
    Submit face verification task (async processing)
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Read images and convert to base64
    cnic_base64 = await read_upload_as_base64(cnic_image)
    selfie_base64 = await read_upload_as_base64(selfie_image)
    
    # Store task metadata
    store_task_metadata(task_id)
    
    # Add to Redis queue
    task_data = {
        "task_id": task_id,
        "type": "verify_face",
        "cnic_base64": cnic_base64,
        "selfie_base64": selfie_base64,
        "webhook_url": f"https://your-railway-app.railway.app/webhook/result/{task_id}",
        "webhook_error_url": f"https://your-railway-app.railway.app/webhook/error/{task_id}"
    }
    
    redis_client.lpush("cnic_tasks", json.dumps(task_data))
    
    logger.info(f"Task {task_id} queued for face verification")
    
    return TaskResponse(
        task_id=task_id,
        status="queued",
        message="Face verification task has been queued"
    )

@app.get("/result/{task_id}", response_model=TaskResultResponse)
async def get_result(task_id: str):
    """
    Get task result
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    task_data = redis_client.get(f"task:{task_id}")
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    data = json.loads(task_data)
    
    return TaskResultResponse(
        task_id=task_id,
        status=data.get("status", "unknown"),
        result=data.get("result"),
        error=data.get("error"),
        created_at=data.get("created_at"),
        completed_at=data.get("completed_at")
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api_light:app", host="0.0.0.0", port=port, reload=False)
