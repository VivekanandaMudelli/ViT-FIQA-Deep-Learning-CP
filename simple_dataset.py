import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

        for label, person in enumerate(os.listdir(root_dir)):
            person_dir = os.path.join(root_dir, person)
            if os.path.isdir(person_dir):
                for img in os.listdir(person_dir):
                    self.image_paths.append(os.path.join(person_dir, img))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # img = Image.open(self.image_paths[idx]).convert("RGB")
        img = Image.open(self.image_paths[idx])

        # convert grayscale → RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        img = self.transform(img)
        label = self.labels[idx]
        return img, label