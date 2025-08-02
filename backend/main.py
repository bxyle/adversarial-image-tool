from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import numpy as np
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Load image
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = np.array(img)

        # Simple manipulation: invert (for placeholder)
        manipulated = 255 - img_array  # invert as a placeholder

        # Save manipulated image
        output_path = "static/output.png"
        Image.fromarray(manipulated).save(output_path)

        # Return success message with download URL
        return JSONResponse(content={"download_url": "/download"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/download")
def download():
    filepath = "static/output.png"
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png", filename="output.png")
    return JSONResponse(content={"error": "File not found"}, status_code=404)
