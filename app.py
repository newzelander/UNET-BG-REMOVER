from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from PIL import Image, UnidentifiedImageError, ImageFile
import io
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
import logging
import gc

# Import U2NET model class
from model.u2net import U2NET

app = FastAPI()

# Allow CORS from Webflow
origins = ["https://spaceluma.webflow.io"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to limit upload size
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_upload_size:
            return Response("File too large", status_code=413)
        return await call_next(request)

app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=20 * 1024 * 1024)  # 20 MB

# Constants
MODEL_PATH = "saved_models/u2net/u2net.pth"
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_DIMENSION = 1024  # Resize images larger than this dimension

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load U2NET model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = U2NET(3, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Resize large images early
def resize_large_image(img):
    max_side = max(img.size)
    if max_side > MAX_DIMENSION:
        scale = MAX_DIMENSION / max_side
        new_size = (int(img.width * scale), int(img.height * scale))
        img.resize(new_size, Image.Resampling.LANCZOS)
    return img

# Preprocess
def preprocess(pil_image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return transform(pil_image).unsqueeze(0)

# Remove background
def remove_background(pil_image):
    input_tensor = preprocess(pil_image).to(device)
    with torch.no_grad():
        d1, *_ = model(input_tensor)
        pred = d1[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred_np = pred.squeeze().cpu().numpy()

    # Free GPU and CPU memory
    del d1, pred, input_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create alpha mask
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
    return Image.fromarray(output_rgba, mode="RGBA")

# Endpoint
@app.post("/remove-background")
async def api_remove_background(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max allowed is {MAX_FILE_SIZE_MB}MB.")

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image format.")
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        raise HTTPException(status_code=500, detail="Error reading image.")
    finally:
        file.file.close()

    try:
        pil_image = resize_large_image(pil_image)
        result_img = remove_background(pil_image)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.exception("Error during background removal")
        raise HTTPException(status_code=500, detail="Background removal failed.")

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}
