# Use a minimal Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone U-2-Net
RUN git clone https://github.com/NathanUA/U-2-Net.git U-2-Net

# Download the model weights
RUN mkdir -p U-2-Net/saved_models/u2net && \
    gdown --id 1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy -O U-2-Net/saved_models/u2net/u2net.pth

# Copy your FastAPI app code
COPY . .

# Expose the port Uvicorn will run on
EXPOSE 8000

# Command to run your app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
