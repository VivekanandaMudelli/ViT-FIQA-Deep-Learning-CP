import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class YaleDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

        for label, person in enumerate(os.listdir(root_dir)):
            person_path = os.path.join(root_dir, person)
            if os.path.isdir(person_path):
                for img in os.listdir(person_path):
                    self.image_paths.append(os.path.join(person_path, img))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label