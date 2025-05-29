# Use an official slim Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Clone U-2-Net repo
RUN git clone https://github.com/NathanUA/U-2-Net.git U-2-Net

# Optional: Download model weights (if you're not downloading at runtime)
# RUN gdown --id 1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy -O U-2-Net/saved_models/u2net/u2net.pth

# Copy your application code
COPY app.py .

# Set PYTHONPATH to allow imports from U-2-Net
ENV PYTHONPATH=/app/U-2-Net

# Expose FastAPI port
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
