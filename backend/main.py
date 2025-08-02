
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image

app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image = Image.open(file.file)

    # Kép mentése memóriába, hogy vissza tudjuk küldeni
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")



from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import json
import requests

app = FastAPI()

# Modell betöltése és osztálynevek lekérése
model = models.resnet50(pretrained=True)
model.eval()

# ImageNet osztálynevek letöltése
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(LABELS_URL)
categories = response.text.strip().split("\n")

# Preprocess
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        out = model(batch_t)
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        predictions = [
            {"label": categories[idx], "confidence": percentage[idx].item()}
            for idx in indices[0][:5]
        ]
    return JSONResponse(content={"predictions": predictions})



import torch.nn as nn

def fgsm_attack(image, epsilon, data_grad):
    # Előjel lekérése a gradiensből
    sign_data_grad = data_grad.sign()
    # Adversarial kép generálása
    perturbed_image = image + epsilon * sign_data_grad
    # Clamp [0,1] tartományba
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

@app.post("/fgsm/")
async def apply_fgsm(file: UploadFile = File(...), epsilon: float = 0.03, target_label: int = -1):
    image = Image.open(file.file).convert("RGB")
    img_t = transform(image)
    img_t.requires_grad = True
    batch_t = torch.unsqueeze(img_t, 0)

    model.zero_grad()
    output = model(batch_t)

    if target_label >= 0:
        loss = nn.CrossEntropyLoss()(output, torch.tensor([target_label]))
    else:
        pred_label = output.max(1, keepdim=True)[1]
        loss = nn.CrossEntropyLoss()(output, pred_label.squeeze())

    loss.backward()
    data_grad = img_t.grad.data

    perturbed_data = fgsm_attack(img_t, epsilon, data_grad)
    adv_img = perturbed_data.squeeze().detach().numpy().transpose(1, 2, 0)
    adv_img = (adv_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    adv_img = (adv_img * 255).clip(0, 255).astype("uint8")
    adv_pil = Image.fromarray(adv_img)

    buf = BytesIO()
    adv_pil.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")



import numpy as np
from PIL import ImageDraw

def add_adversarial_patch(image: Image.Image, patch_size: int = 50, position: str = "bottom_right"):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Choose position
    if position == "top_left":
        x, y = 0, 0
    elif position == "top_right":
        x, y = width - patch_size, 0
    elif position == "bottom_left":
        x, y = 0, height - patch_size
    else:  # bottom_right
        x, y = width - patch_size, height - patch_size

    # Random patch generation (could be optimized later using adversarial learning)
    patch = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
    patch_img = Image.fromarray(patch, mode='RGB')
    image.paste(patch_img, (x, y))

    return image

@app.post("/adv-patch/")
async def apply_adversarial_patch(file: UploadFile = File(...), patch_size: int = 50, position: str = "bottom_right"):
    image = Image.open(file.file).convert("RGB")
    patched = add_adversarial_patch(image, patch_size=patch_size, position=position)

    buf = BytesIO()
    patched.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")



def add_gaussian_noise(image: Image.Image, mean: float = 0.0, stddev: float = 25.0):
    np_image = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, stddev, np_image.shape)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

@app.post("/gaussian-noise/")
async def apply_gaussian_noise(file: UploadFile = File(...), mean: float = 0.0, stddev: float = 25.0):
    image = Image.open(file.file).convert("RGB")
    noisy = add_gaussian_noise(image, mean=mean, stddev=stddev)

    buf = BytesIO()
    noisy.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")



from PIL import ImageEnhance

def manipulate_colors(image: Image.Image, contrast: float = 1.0, gamma: float = 1.0, red_shift: int = 0, green_shift: int = 0, blue_shift: int = 0):
    # Kontraszt állítás
    image = ImageEnhance.Contrast(image).enhance(contrast)

    # Gamma korrekció
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    image = image.point(table * 3)

    # Színcsatornák eltolása
    r, g, b = image.split()
    r = r.point(lambda i: min(max(i + red_shift, 0), 255))
    g = g.point(lambda i: min(max(i + green_shift, 0), 255))
    b = b.point(lambda i: min(max(i + blue_shift, 0), 255))
    image = Image.merge("RGB", (r, g, b))

    return image

@app.post("/color-manip/")
async def apply_color_manipulations(
    file: UploadFile = File(...),
    contrast: float = 1.0,
    gamma: float = 1.0,
    red_shift: int = 0,
    green_shift: int = 0,
    blue_shift: int = 0
):
    image = Image.open(file.file).convert("RGB")
    modified = manipulate_colors(image, contrast, gamma, red_shift, green_shift, blue_shift)

    buf = BytesIO()
    modified.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")



import random
import cairosvg
from PIL import ImageOps

def generate_svg_overlay(width: int, height: int, density: int = 10):
    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    for _ in range(density):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(5, 20)
        svg += f'<circle cx="{x}" cy="{y}" r="{r}" fill="rgba(255,255,255,0.05)"/>'
    svg += '</svg>'
    return svg

def overlay_svg_on_image(image: Image.Image, density: int = 10):
    width, height = image.size
    svg_data = generate_svg_overlay(width, height, density)
    png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), output_width=width, output_height=height)
    overlay = Image.open(BytesIO(png_data)).convert("RGBA")

    # Konvertáljuk az eredeti képet RGBA formátumra és összefésüljük
    base = image.convert("RGBA")
    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")

@app.post("/svg-overlay/")
async def apply_svg_overlay(file: UploadFile = File(...), density: int = 10):
    image = Image.open(file.file).convert("RGB")
    overlayed = overlay_svg_on_image(image, density=density)

    buf = BytesIO()
    overlayed.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")



@app.post("/spoof/")
async def spoof_target_class(file: UploadFile = File(...), target_class: str = "golden retriever", epsilon: float = 0.03):
    image = Image.open(file.file).convert("RGB")
    img_t = transform(image)
    img_t.requires_grad = True
    batch_t = torch.unsqueeze(img_t, 0)

    # Get index of target class
    try:
        target_index = categories.index(target_class.lower())
    except ValueError:
        return JSONResponse(content={"error": "Invalid target_class name."}, status_code=400)

    model.zero_grad()
    output = model(batch_t)

    loss = nn.CrossEntropyLoss()(output, torch.tensor([target_index]))
    loss.backward()
    data_grad = img_t.grad.data

    perturbed_data = fgsm_attack(img_t, epsilon, data_grad)
    adv_img = perturbed_data.squeeze().detach().numpy().transpose(1, 2, 0)
    adv_img = (adv_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    adv_img = (adv_img * 255).clip(0, 255).astype("uint8")
    adv_pil = Image.fromarray(adv_img)

    buf = BytesIO()
    adv_pil.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
