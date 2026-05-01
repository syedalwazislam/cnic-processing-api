FROM python:3.9-slim-bullseye
# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir cmake==3.25.0
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu
RUN CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install --no-cache-dir dlib==19.24.2
# Install other requirements
RUN pip install --no-cache-dir \
    redis==5.0.1 \
    ultralytics==8.0.207 \
    opencv-python-headless==4.8.1.78 \
    easyocr==1.7.1 \
    numpy==1.24.3 \
    requests==2.31.0 \
    arabic-reshaper==3.0.0 \
    python-bidi==0.4.2 \
    pandas==2.0.3 \
    face-recognition==1.3.0 \
    google-genai>=0.5.0 \
    pillow>=9.0.0
# Copy application files
COPY worker.py .
COPY webtest.py .
# ===== ADD THESE LINES TO COPY YOUR MODEL =====
# Create the model directory structure
RUN mkdir -p runs/detect/train3/weights/
# Copy your model file (adjust the source path if needed)
COPY runs/detect/train3/weights/best.pt runs/detect/train3/weights/best.pt
# Verify
RUN python -c "import face_recognition; print('✅ face_recognition working!')"
CMD ["python", "-u", "worker.py"]