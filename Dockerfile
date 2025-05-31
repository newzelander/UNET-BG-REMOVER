# Use minimal python image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for your app and U-2-Net
RUN apt-get update && apt-get install -y \
    git wget curl libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Clone U-2-Net repository (needed for model and code)
RUN git clone https://github.com/NathanUA/U-2-Net.git U-2-Net

# Download the U-2-Net weights to the right folder
RUN mkdir -p U-2-Net/saved_models/u2net && \
    curl -L -o U-2-Net/saved_models/u2net/u2net.pth "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy"

# Copy your application code into the image
COPY . .

# Expose port 8000 (for uvicorn)
EXPOSE 8000

# Start uvicorn server, listening on all interfaces (required by Fly.io)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
