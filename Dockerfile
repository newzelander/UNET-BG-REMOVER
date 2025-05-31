FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    opencv-python \
    numpy \
    torch \
    torchvision \
    fastapi \
    uvicorn \
    gdown \
    pillow

# Clone the U-2-Net repository
RUN git clone https://github.com/NathanUA/U-2-Net.git U-2-Net

# Create the weights directory
RUN mkdir -p /app/U-2-Net/saved_models/u2net

# Download the model weight file from Google Drive
RUN gdown https://drive.google.com/uc?id=1HlUE4ZblTppQfQ2HwqOUYx6RX_V3TRKF -O /app/U-2-Net/saved_models/u2net/u2net.pth

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
