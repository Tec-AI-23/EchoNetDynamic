from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


def generate_data_dict(data, filename):
    return {"data": data, "filename": filename}


class EchoDatasetMasks(Dataset):
    def __init__(self, images_paths, masks_paths, transform=None):
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        img_path = self.images_paths[index]
        mask_path = self.masks_paths[index]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        mask[mask == 255.0] = 1.0

        image_data = generate_data_dict(image, img_path)
        mask_data = generate_data_dict(mask, mask_path)

        return {"image": image_data, "mask": mask_data}


class EchoDatasetHeatmaps(Dataset):
    def __init__(self, images_paths, heatmaps_paths, masks_paths, transform=None):
        self.images_paths = images_paths
        self.heatmaps_paths = heatmaps_paths
        self.masks_paths = masks_paths
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        img_path = self.images_paths[index]
        mask_path = self.masks_paths[index]
        heatmap_path = self.heatmaps_paths[index]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        heatmap = torch.load(heatmap_path)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        mask[mask == 255.0] = 1.0

        image_data = generate_data_dict(image, img_path)
        mask_data = generate_data_dict(mask, mask_path)
        heatmap_data = generate_data_dict(heatmap, heatmap_path)
        return {"image": image_data, "heatmap": heatmap_data, "mask": mask_data}
