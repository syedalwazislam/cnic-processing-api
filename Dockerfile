FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY api/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire api folder
COPY api/ .

# Set environment variable
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Run the application
CMD python -m uvicorn api:app --host 0.0.0.0 --port $PORT