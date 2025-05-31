FROM python:3.9-slim

# Install system dependencies and gdown
RUN apt-get update && apt-get install -y git wget && pip install --no-cache-dir gdown && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone U-2-Net repo to get model code files
RUN git clone https://github.com/NathanUA/U-2-Net.git

# Copy only model files (including subfolders like __pycache__)
RUN mkdir model && cp -r U-2-Net/model/* model/

# Download pretrained weights
RUN mkdir -p saved_models/u2net && \
    gdown --id 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ -O saved_models/u2net/u2net.pth

# Clean up
RUN rm -rf U-2-Net

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose port
EXPOSE 8080

# Add healthcheck script (optional, for local container testing)
RUN echo '#!/bin/sh\ncurl -f http://localhost:8080/health || exit 1' > /healthcheck.sh && chmod +x /healthcheck.sh

# Run FastAPI app with uvicorn with debug logs
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8080 --log-level debug"]
