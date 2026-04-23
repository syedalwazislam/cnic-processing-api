FROM python:3.9-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ .

# Use environment variable with default
ENV PORT=8000

# Run with the environment variable
CMD sh -c "uvicorn api:app --host 0.0.0.0 --port $PORT"