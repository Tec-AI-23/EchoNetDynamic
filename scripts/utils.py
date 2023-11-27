import matplotlib.pyplot as plt
import torch
import torchvision
from dataset import EchoDatasetHeatmaps, EchoDatasetMasks
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
from torchmetrics.classification import Dice
from tqdm import tqdm


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders_masks(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = EchoDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = EchoDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_loaders_landmarks():
    pass


def get_loaders_landmarks(
    train_dir,
    train_maskdir,
    train_heatmaps,
    val_dir,
    val_maskdir,
    val_heatmaps,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = EchoDatasetHeatmap(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        heatmap_dir=train_heatmaps,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = EchoDatasetHeatmap(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        heatmap_dir=val_heatmaps,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, model_type, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    if model_type == "masks":
        with torch.no_grad():
            for data_dir in loader:
                x = data_dir["image"]["data"]
                y = data_dir["mask"]["data"]
                
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice = Dice().to(device)
                preds, y = preds.int(), y.int()
                dice_score += dice(preds, y)

        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
        )
        print(f"Dice score: {dice_score/len(loader)*100:.2f}")
        model.train()

    elif model_type == "landmarks":
        with torch.no_grad():
            for x, z, y in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)

                preds = model(x)
                # convert heatmap to image
                preds = heatmap_to_image(preds).to(device)
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice = Dice().to(device)
                preds, y = preds.int(), y.int()
                dice_score += dice(preds, y)

        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
        )
        print(f"Dice score: {dice_score/len(loader)*100:.2f}")
        model.train()


def folder_creation(folder):
    isExist = os.path.exists(folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(folder)


def save_predictions_as_imgs(
    loader, model, model_type, folder="saved_images/", device="cuda"
):
    folder_creation(folder)
    model.eval()
    if model_type == "masks":
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds >= 0.5).float()
                # save masks predictions as image
                folder_creation(f"{folder}/masks_predictions")
                for prediction in range(preds.shape[0]):
                    torchvision.utils.save_image(
                        preds[prediction],
                        f"{folder}/masks_predictions/batch_{idx}_no{prediction}.png",
                    )

                # Saving original masks as individual images
                folder_creation(f"{folder}/original_masks")

                y[y == 1] = 255
                for mask in range(y.shape[0]):
                    torchvision.utils.save_image(
                        y[mask], f"{folder}/original_masks/batch{idx}_no{mask}.png"
                    )

                    # cv2.imwrite(f"{folder}/original_masks/batch{idx}_no{mask}.png", y[mask].numpy())

                # #Saving the masks as batch
                # torchvision.utils.save_image(y, f"{folder}/{idx}.png")

    elif model_type == "landmarks":
        for idx, (x, z, y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
        # save heatmaps predictions as images, individually
        save_batch_to_coordinate(preds, idx, folder)

        # Saving original masks as individual images
        folder_creation(f"{folder}/original_masks")

        y[y == 1] = 255
        for mask in range(y.shape[0]):
            torchvision.utils.save_image(
                y[mask], f"{folder}/original_masks/batch{idx}_no{mask}.png"
            )

        # #Saving original images as individual images
        # for imagen in range(y.shape[0]):
        #     #cv2.imwrite(f"{folder}/image{idx}_{imagen}.png", x[imagen].permute(1,2,0).to("cpu").numpy())
        #     torchvision.utils.save_image(x[imagen], f"{folder}/image{idx}_{imagen}.png")

        # saving original images as batch
        # torchvision.utils.save_image(x, f"{folder}/image{idx}.png")

    model.train()


def unravel_index(indices, shape):
    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


def save_batch_to_coordinate(t, idx, folder):
    for batch in range(t.shape[0]):
        img = np.zeros(t.shape[2:], dtype=np.uint8)
        coor = []
        for i in range(t.shape[1]):
            coordenada = unravel_index(t[batch][i].argmax(), t.shape[2:])
            coor.append(coordenada)
        points = np.array(coor)
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_points = points[np.argsort(angles)]
        poly = np.array(sorted_points, np.int32)
        cv2.fillPoly(img, [poly], 255)

        # guardar
        folder_creation(f"{folder}/landmark_predictions")
        cv2.imwrite(f"{folder}/landmark_predictions/prueba_batch{idx}_{batch}.png", img)


def heatmap_to_image(t):
    masks = torch.tensor([])
    for batch in range(t.shape[0]):
        img = np.zeros(t.shape[2:], dtype=np.uint8)
        coor = []
        for i in range(t.shape[1]):
            coordenada = unravel_index(t[batch][i].argmax(), t.shape[2:])
            coor.append(coordenada)
        points = np.array(coor)
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_points = points[np.argsort(angles)]
        poly = np.array(sorted_points, np.int32)
        cv2.fillPoly(img, [poly], 255)

        mask = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        masks = torch.cat((masks, mask))
    return masks


def train_fn(loader, model, optimizer, loss_fn, scaler, DEVICE, model_type):
    loop = tqdm(loader)

    if model_type == "masks":
        for batch_idx, data_dir in enumerate(loop):
            data = data_dir["image"]["data"]
            data = data.to(device=DEVICE)

            targets = data_dir["mask"]["data"]
            targets = targets.float().to(device=DEVICE)

            # if model is mask, target need color dimension
            if len(targets.shape) != len(data.shape):
                targets = targets.unsqueeze(1)

            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)

                loss = loss_fn(predictions, targets)
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

    elif model_type == "landmarks":
        for batch_idx, (data, targets, masks) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.float().to(device=DEVICE)

            # if model is mask, target need color dimension
            if len(targets.shape) != len(data.shape):
                targets = targets.unsqueeze(1)

            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)

                loss = loss_fn(predictions, targets)
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
