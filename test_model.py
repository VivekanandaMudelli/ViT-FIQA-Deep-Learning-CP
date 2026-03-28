import torch
from backbones.vit_qs import VisionTransformer

model = VisionTransformer(
    img_size=112,
    patch_size=9,
    num_classes=512,
    embed_dim=512,
    depth=12,
    num_heads=8,
    drop_path_rate=0.1,
    norm_layer="ln",
    mask_ratio=0.1
)

model.eval()

x = torch.randn(1, 3, 112, 112)

out = model(x)

embedding, quality = model(x)

print("Quality score:", quality.item())