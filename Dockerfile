FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install gdown
RUN pip install --no-cache-dir gdown

# Set working directory
WORKDIR /app

# Clone U-2-Net repository
RUN git clone https://github.com/NathanUA/U-2-Net.git

# Copy model files
RUN mkdir model && cp -r U-2-Net/model/* model/

# Download pretrained weights
RUN mkdir -p saved_models/u2net && \
    gdown --id 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ -O saved_models/u2net/u2net.pth

# Clean up repository
RUN rm -rf U-2-Net

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8080

# Healthcheck script
RUN echo '#!/bin/sh\ncurl -f http://localhost:8080/health || exit 1' > /healthcheck.sh && chmod +x /healthcheck.sh

# Run the application
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8080 --log-level debug"]
