import sys
import os
import io
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import torchvision.transforms as transforms

# Add U-2-Net repo to path
sys.path.append("/app/U-2-Net")

from model.u2net import U2NET

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_weights_if_needed():
    import gdown
    os.makedirs("/app/U-2-Net/saved_models/u2net", exist_ok=True)
    url = "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
    output_path = "/app/U-2-Net/saved_models/u2net/u2net.pth"
    if not os.path.exists(output_path):
        print("Downloading U2NET weights...")
        gdown.download(url, output_path, quiet=False)
        print("Download complete.")
    else:
        print("Weights already exist.")

def load_model():
    download_weights_if_needed()
    model = U2NET(3, 1)
    model.load_state_dict(torch.load("/app/U-2-Net/saved_models/u2net/u2net.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ U^2-Net model loaded successfully")
    return model

model = load_model()

def preprocess_image(pil_image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    return transform(pil_image).unsqueeze(0)

@app.get("/")
async def root():
    return {"message": "U-2-Net Background Remover API running"}

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    if file.content_type.split('/')[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size

    input_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        d1, *_ = model(input_tensor)
        pred = d1[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        pred_np = pred.squeeze().cpu().numpy()

    # Resize mask to original size
    alpha = cv2.resize(pred_np, (original_size[0], original_size[1]), interpolation=cv2.INTER_LINEAR)

    # Blend with slight blur (suppress halo)
    alpha_blur = cv2.GaussianBlur(alpha, (5, 5), sigmaX=1.2)
    alpha_mix = np.clip(alpha * 0.75 + alpha_blur * 0.25, 0, 1)

    # Boost transparency in semi-edge zones
    alpha_boosted = np.power(alpha_mix, 0.85)

    # Morphological inward trim (deeper background cleanup)
    mask_uint8 = (alpha_boosted * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_eroded = cv2.erode(mask_uint8, kernel, iterations=2)
    alpha_final = mask_eroded.astype(np.float32) / 255.0

    # Compose final RGBA image
    img_np = np.array(image).astype(np.uint8)
    output_rgba = np.dstack((img_np, (alpha_final * 255).astype(np.uint8)))

    result = Image.fromarray(output_rgba, mode="RGBA")

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
