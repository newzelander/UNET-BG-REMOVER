# Use an official Python runtime as a parent image 
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone the U-2-Net repository
RUN git clone https://github.com/NathanUA/U-2-Net.git U-2-Net

# Install gdown and download the model weights
RUN pip install --no-cache-dir gdown && \
    mkdir -p U-2-Net/saved_models/u2net && \
    gdown --id 1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy -O U-2-Net/saved_models/u2net/u2net.pth

# Optionally install other Python dependencies
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# Optionally copy your app code
# COPY . .

# Set default command
# CMD ["python", "your_app.py"]
