# Use minimal python image
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git wget curl libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Clone U-2-Net repo
RUN git clone https://github.com/NathanUA/U-2-Net.git U-2-Net

# Download weights (optional: you can keep it here or do in app.py)
RUN mkdir -p U-2-Net/saved_models/u2net && \
    curl -L -o U-2-Net/saved_models/u2net/u2net.pth "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
