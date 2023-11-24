import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from heatmap_creation import(
    create_contour,create_gaussian_heatmaps,create_countour_heatmaps
)

class EchoDatasetHeatmap(Dataset):
    def __init__(self, image_dir, mask_dir,csv_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.df_coor = pd.read_csv(csv_file)
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        img_name = self.images[index].partition("_")[0]+'.avi'
        img_frame = self.images[index].partition("_")[2].partition(".")[0]

        coor = self.df_coor[(self.df_coor.FileName == img_name) & (self.df_coor.Frame == int(img_frame))]
        puntos1 = [(int(row['X1']), int(row['Y1'])) for index, row in coor.iterrows()]
        puntos2 = [(int(row['X2']), int(row['Y2'])) for index, row in coor.iterrows()]
        puntos1 = [puntos1[0], puntos1[5], puntos1[10], puntos1[15]]
        puntos2 = [puntos2[0], puntos2[5], puntos2[10], puntos2[15]]
        #puntos2 = puntos2[::6]

        poly = np.array(np.concatenate((puntos1, puntos2), axis=0).tolist(), np.int32)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)


        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        heatmap = create_countour_heatmaps(mask, poly)
        mask[mask == 255.0] = 1.0


        return image, heatmap, mask
