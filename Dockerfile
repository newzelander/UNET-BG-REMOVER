FROM python:3.9-slim

# Install system dependencies, including libgl1-mesa-glx and libglib2.0-0 + libglib2.0-bin for libgthread-2.0.so.0, plus gdown
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      wget \
      curl \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libglib2.0-bin && \
    pip install --no-cache-dir gdown && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone U-2-Net repo to get model code files
RUN git clone https://github.com/NathanUA/U-2-Net.git

# Copy only model files (including subfolders like __pycache__)
RUN mkdir model && cp -r U-2-Net/model/* model/

# Download pretrained weights
RUN mkdir -p saved_models/u2net && \
    gdown --id 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ -O saved_models/u2net/u2net.pth

# Clean up repo folder
RUN rm -rf U-2-Net

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose port 8080
EXPOSE 8080

# Optional healthcheck script for local container testing
RUN echo '#!/bin/sh\ncurl -f http://localhost:8080/health || exit 1' > /healthcheck.sh && chmod +x /healthcheck.sh

# Run FastAPI app with uvicorn and debug logs
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8080 --log-level debug"]
