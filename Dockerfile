# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && apt-get clean

# Clone the U-2-Net repository
RUN git clone https://github.com/NathanUA/U-2-Net.git U-2-Net

# Install gdown and download the model weights, creating required directories first
RUN pip install --no-cache-dir gdown && \
    mkdir -p U-2-Net/saved_models/u2net && \
    gdown --id 1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy -O U-2-Net/saved_models/u2net/u2net.pth

# Install other python dependencies if you have requirements.txt
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code (adjust as needed)
# COPY . .

# Expose any ports if necessary
# EXPOSE 8000

# Set default command
# CMD ["python", "your_app.py"]
