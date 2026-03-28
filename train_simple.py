import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backbones.vit_qs import VisionTransformer
from simple_dataset import FaceDataset

# ===== MODEL =====
model = VisionTransformer(
    img_size=112,
    patch_size=9,
    num_classes=512,
    embed_dim=256,
    depth=6,
    num_heads=4,
    drop_path_rate=0.1,
    norm_layer="ln",
    mask_ratio=0.1
)

# ===== DATA =====
dataset = FaceDataset(r"C:\Users\mudel\Desktop\SEM_6\DL\Project\ViT-FIQA-Assessing-Face-Image-Quality-using-Vision-Transformers\cropped")


imagecount = 17000
b_size = 10


loader = DataLoader(dataset, batch_size=b_size, shuffle=True)

# ===== LOSSES =====
classifier = nn.Linear(256, len(set(dataset.labels)))
ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()

# ===== OPTIMIZER =====
optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=1e-4)

# ===== TRAIN LOOP =====
model.train()

for epoch in range(20):
    total_loss = 0

    for i, (imgs, labels) in enumerate(loader):
        embeddings, quality = model(imgs)

        # classification loss
        logits = classifier(embeddings)
        loss_cls = ce_loss(logits, labels)

        # fake quality target (temporary)
        # quality_target = torch.rand_like(quality)

        prob = torch.softmax(logits, dim=1)
        confidence = prob.max(dim=1)[0].unsqueeze(1)
        loss_q = mse_loss(quality, confidence.detach())

        loss = loss_cls + loss_q

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

    total_loss /= imagecount
    total_loss *= b_size
    print(f"Epoch {epoch}: Loss = {total_loss:.4f}")