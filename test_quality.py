import os
import torch
from PIL import Image, ImageFilter
import torchvision.transforms as transforms

from backbones.vit_qs import VisionTransformer

# ===== MODEL =====
model = VisionTransformer(
    img_size=112,
    patch_size=9,
    num_classes=512,
    embed_dim=128,
    depth=4,
    num_heads=2,
    drop_path_rate=0.1,
    norm_layer="ln",
    mask_ratio=0.1
)

model.eval()

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

# ===== PATH TO YALE DATASET =====
root = "C:/Users/mudel/Desktop/SEM_6/DL/Project/ViT-FIQA-Assessing-Face-Image-Quality-using-Vision-Transformers/cropped"

# pick one folder
person = os.listdir(root)[0]
person_path = os.path.join(root, person)

# pick one image
img_name = os.listdir(person_path)[0]
img_path = os.path.join(person_path, img_name)

print("Testing image:", img_path)

# ===== LOAD IMAGE =====
img = Image.open(img_path)

if img.mode != "RGB":
    img = img.convert("RGB")

img = transform(img).unsqueeze(0)

# ===== PREDICT =====
with torch.no_grad():
    embedding, quality = model(img)

print("Quality score (original):", quality.item())

# ===== TEST BLUR =====
blurred = img.squeeze(0)
blurred = transforms.ToPILImage()(blurred)
blurred = blurred.filter(ImageFilter.GaussianBlur(5))
blurred = transform(blurred).unsqueeze(0)

with torch.no_grad():
    _, q_blur = model(blurred)

print("Quality score (blurred):", q_blur.item())