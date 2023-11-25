import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class EchoDatasetMasks(Dataset):
    def __init__(self, images_paths, masks_paths, transform=None):
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        img_path = self.images_paths[index]
        masks_path = self.masks_paths[index]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(masks_path), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class EchoDatasetHeatmap(Dataset):
    def __init__(self, images_paths, heatmaps_dir, transform=None):
        self.images_paths = images_paths
        self.heatmaps_dir = heatmaps_dir
        self.images = os.listdir(images_paths)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_paths, self.images[index])
        heatmaps_path = os.path.join(self.heatmaps_dir, self.images[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        heatmap = torch.load(heatmaps_path)

        if self.transform is not None:
            augmentations = self.transform(image=image, heatmap=heatmap)
            image = augmentations["image"]
            heatmap = augmentations["heatmap"]

        return image, heatmap
