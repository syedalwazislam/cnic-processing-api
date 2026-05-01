FROM python:3.9-slim

# Install system dependencies for OpenCV and face_recognition
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY worker.py .
COPY webtest.py .

# Create directory for model
RUN mkdir -p runs/detect/train3/weights/

# Download default YOLO model (optional, for fallback)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Run the worker
CMD ["python", "-u", "worker.py"]