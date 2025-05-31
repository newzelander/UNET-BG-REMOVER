FROM python:3.9

# Install system packages required by OpenCV and U-2-Net
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Clone U-2-Net repository and extract the model code only
RUN git clone https://github.com/NathanUA/U-2-Net.git && \
    mkdir model && cp -r U-2-Net/model/* model/ && \
    rm -rf U-2-Net

# Download pretrained model weights
RUN mkdir -p saved_models/u2net && \
    pip install --no-cache-dir gdown && \
    gdown --id 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ -O saved_models/u2net/u2net.pth

# Copy Python dependencies
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose FastAPI port
EXPOSE 8080

# Optional healthcheck script
RUN echo '#!/bin/sh\ncurl -f http://localhost:8080/health || exit 1' > /healthcheck.sh && chmod +x /healthcheck.sh

# Set environment variable for library path
ENV LD_LIBRARY_PATH=/lib/x86_64-linux-gnu

# Start FastAPI server
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8080 --log-level debug"]
