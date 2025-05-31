FROM python:3.9-slim

# Install system dependencies: git, wget, and gdown
RUN apt-get update && apt-get install -y git wget && pip install --no-cache-dir gdown && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone U-2-Net repo to get model code files
RUN git clone https://github.com/NathanUA/U-2-Net.git

# Download pretrained weights from Google Drive using gdown
RUN mkdir -p saved_models/u2net \
    && gdown --id 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ -O saved_models/u2net/u2net.pth

# Copy only the model files to separate folder
RUN mkdir model && cp U-2-Net/model/* model/

# Remove cloned repo to keep image clean
RUN rm -rf U-2-Net

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app source code
COPY app.py .

# Expose port (Fly.io or local testing)
EXPOSE 8080

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
