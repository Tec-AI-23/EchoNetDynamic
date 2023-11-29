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


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, model_type, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    if model_type == "masks":
        with torch.no_grad():
            for data_dict in loader:
                x = data_dict["image"]["data"]
                y = data_dict["mask"]["data"]

                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds >= 0.5).float()
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
            for data_dict in loader:
                x = data_dict["image"]["data"]
                z = data_dict["heatmap"]["data"]
                y = data_dict["mask"]["data"]

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
    loader, model, model_type="masks", folder="saved_images/", device="cuda"
):
    folder_creation(folder)
    model.eval()
    if model_type == "masks":
        for idx, data_dict in enumerate(loader):
            x = data_dict["image"]["data"]
            y = data_dict["mask"]["data"]

            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds >= 0.5).float()
                # save masks predictions as image
                folder_creation(f"{folder}/masks_predictions")

                torchvision.utils.save_image(preds, f"{folder}/masks_predictions/batch_{idx}.png")

                # for prediction in range(preds.shape[0]):
                #     torchvision.utils.save_image(
                #         preds[prediction],
                #         f"{folder}/masks_predictions/batch_{idx}_no{prediction}.png",
                #     )

    elif model_type == "landmarks":
        for idx, data_dict in enumerate(loader):
            x = data_dict["image"]["data"]
            z = data_dict["heatmap"]["data"]
            y = data_dict["mask"]["data"]

            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))

            for index, batch in enumerate(preds):
                coordinates = []
                for channel in batch:
                    channel = channel.to('cpu').detach().numpy()
                    max_coordinates = np.unravel_index(np.argmax(channel), channel.shape)
                    coordinates.append(max_coordinates)
                mask = generate_mask_from_coordinates(coordinates, preds.shape[2:])

                folder_creation(f"{folder}/landmark_predictions")
                torchvision.utils.save_image(mask,f"{folder}/landmark_predictions/prueba_batch{idx}_{batch}.png")
 

    model.train()


def generate_mask_from_coordinates(coordinates, shape):
    centroid = np.mean(coordinates, axis=0)
    poly = np.array(coordinates, np.int32)
    img = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(img, [poly], 255)
    img = torch.from_numpy(img)
    return img


def heatmap_to_image(heatmap):
    masks = torch.tensor([])
    for batch in heatmap:
        coordinates = []
        for channel in batch:
            channel = channel.to('cpu').detach().numpy()
            max_coordinates = np.unravel_index(np.argmax(channel), channel.shape)
            coordinates.append(max_coordinates)
        mask = generate_mask_from_coordinates(coordinates, heatmap.shape[2:])
        masks = torch.cat((masks, mask))
    return masks


def train_fn(loader, model, optimizer, loss_fn, scaler, DEVICE, model_type):
    loop = tqdm(loader)

    if model_type == "masks":
        for batch_idx, data_dict in enumerate(loop):
            data = data_dict["image"]["data"]
            data = data.to(device=DEVICE)

            targets = data_dict["mask"]["data"]
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
        for batch_idx, data_dict in enumerate(loop):
            data = data_dict["image"]["data"]
            targets = data_dict["heatmap"]["data"]

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