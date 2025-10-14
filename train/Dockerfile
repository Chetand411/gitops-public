# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Set working directory inside image (not overlapping Git volume)
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the training script inside the image
COPY train_and_push.py .

# Optional: create directories for data, output, and manifests
RUN mkdir -p /workspace/data /workspace/output /workspace/manifests

# Set entrypoint to run the training script
ENTRYPOINT ["python3", "/app/train_and_push.py"]
