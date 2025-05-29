FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including git, wget, libglib, etc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install gdown for weights download
RUN pip install --no-cache-dir gdown

# Clone U-2-Net repo
RUN git clone https://github.com/NathanUA/U-2-Net.git U-2-Net

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app.py
COPY app.py .

# Set PYTHONPATH so python can find U-2-Net modules
ENV PYTHONPATH=/app/U-2-Net

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
