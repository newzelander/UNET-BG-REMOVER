# Use an official slim Python image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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

# Clone U-2-Net repo (you may want to pin to a specific commit or tag for stability)
RUN git clone https://github.com/NathanUA/U-2-Net.git U-2-Net

# Optional: Download model weights during build (requires gdown, or use curl/wget)
# RUN pip install gdown && \
#     gdown --id 1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy -O U-2-Net/saved_models/u2net/u2net.pth

# Copy your application code
COPY app.py .

# Set PYTHONPATH so your app can import from U-2-Net
ENV PYTHONPATH="${PYTHONPATH}:/app/U-2-Net"

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
