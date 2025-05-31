from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
import os

# Import model class from local model files
from model.u2net import U2NET

app = FastAPI()

# Allow your frontend domain for CORS
origins = [
    "https://spaceluma.webflow.io",  # change to your actual frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "saved_models/u2net/u2net.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = U2NET(3, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def preprocess(pil_image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return transform(pil_image).unsqueeze(0)

def remove_background(pil_image):
    input_tensor = preprocess(pil_image).to(device)
    with torch.no_grad():
        d1, *_ = model(input_tensor)
        pred = d1[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        pred_np = pred.squeeze().cpu().numpy()

    alpha = cv2.resize(pred_np, (pil_image.width, pil_image.height), interpolation=cv2.INTER_LINEAR)

    alpha_blur = cv2.GaussianBlur(alpha, (5, 5), sigmaX=1.2)
    alpha_mix = np.clip(alpha * 0.75 + alpha_blur * 0.25, 0, 1)
    alpha_boosted = np.power(alpha_mix, 0.85)
    mask_uint8 = (alpha_boosted * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_eroded = cv2.erode(mask_uint8, kernel, iterations=2)
    alpha_final = mask_eroded.astype(np.float32) / 255.0

    img_np = np.array(pil_image).astype(np.uint8)
    output_rgba = np.dstack((img_np, (alpha_final * 255).astype(np.uint8)))

    img_rgba = Image.fromarray(output_rgba, mode="RGBA")
    return img_rgba

@app.post("/remove-background")
async def api_remove_background(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    output_img = remove_background(pil_image)

    buf = io.BytesIO()
    output_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
