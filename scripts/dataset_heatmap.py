import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class EchoDatasetHeatmap(Dataset):
    def __init__(self, image_dir, mask_dir,heatmap_dir, transform=None, heatmap_type = 'gaussian'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.heatmap_dir =heatmap_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.heatmap_type = '/'+heatmap_type
        self.heatmaps = os.listdir(heatmap_dir+self.heatmap_type)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        heatmap_dir = os.path.join(self.heatmap_dir+self.heatmap_type, self.heatmaps[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        heatmap = torch.load(heatmap_dir)


        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        mask[mask == 255.0] = 1.0

        return image, heatmap, mask
