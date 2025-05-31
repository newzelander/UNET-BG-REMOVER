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

RUN git clone https://github.com/NathanUA/U-2-Net.git U-2-Net

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
