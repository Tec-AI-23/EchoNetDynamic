import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.classification import Dice
from utils import save_predictions_as_imgs
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torchvision

def folder_creation(folder):
    isExist = os.path.exists(folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(folder)

def sort_coordinates(coordinates):
    coordinates = np.array(coordinates)
    centroid = np.mean(coordinates, axis=0)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    angles = np.arctan2(y - centroid[1], x - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_points = [coordinates[i] for i in sorted_indices]
    return sorted_points

def generate_mask_from_coordinates(coor, shape):
    coordinates = sort_coordinates(coor)
    poly = np.array(coordinates, np.int32)
    img = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(img, [poly], 255)
    return img


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_dice_score = 0, 0
    
    # Loop through data loader data batches
    for index_batch, (data_dict) in enumerate(dataloader):
        # Send data to target device
        X = data_dict["image"]["data"]
        y = data_dict["heatmap"]["data"]
        X, y = X.to(device), y.to(device)

        z  = data_dict["mask"]["data"]

        # 1. Forward pass
        y_pred = model(X).float()


        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y.float())
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate dice score metric across all batches

        # Find the coordinates of the maximum value, per channel

        dice = Dice().to(device)
        y_pred = torch.sigmoid(y_pred)

        batch_train_dice_score = 0
        for index, batch in enumerate(y_pred):
            coordinates = []
            for channel in batch:
                channel = channel.to('cpu').detach().numpy()
                max_coordinates = np.unravel_index(np.argmax(channel), channel.shape)
                coordinates.append(max_coordinates)

            mask = generate_mask_from_coordinates(coordinates, y_pred.shape[2:])
            mask[mask == 255.0] = 1.0

            mask= torch.tensor(mask)
            batch_train_dice_score += dice(z[index].int(), mask.int())

        train_dice_score = batch_train_dice_score / y_pred.shape[0]

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_dice_score = train_dice_score / len(dataloader)
    return train_loss, train_dice_score


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device, epoch):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_dice_score = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for index_batch, (data_dict) in enumerate(dataloader):
            X = data_dict["image"]["data"]
            y = data_dict["heatmap"]["data"]
            # Send data to target device
            X, y = X.to(device), y.to(device)

            z  = data_dict["mask"]["data"]
            # 1. Forward pass
            y_pred = model(X).float()
            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y.float())
            test_loss += loss.item()
            
            # Calculate and accumulate dice score metric across all batches
           
            # Find the coordinates of the maximum value, per channel

            dice = Dice().to(device)
            y_pred = torch.sigmoid(y_pred)

            folder=f"../saved_images/epoch_{epoch}"
            folder_creation(f"{folder}/heatmap_predictions")
            torch.save(y_pred, f"{folder}/heatmap_predictions/prueba_batch{index_batch}.pt")

            batch_test_dice_score = 0
            for index, batch in enumerate(y_pred):
                coordinates = []
                for channel in batch:
                    channel = channel.to('cpu')
                    max_coordinates = np.unravel_index(np.argmax(channel), channel.shape)
                    coordinates.append(max_coordinates)
                mask = generate_mask_from_coordinates(coordinates, y_pred.shape[2:])
                mask[mask == 255.0] = 1.0
                mask= torch.tensor(mask)

                #guardar mascara
                folder=f"../saved_images/epoch_{epoch}"
                folder_creation(f"{folder}/landmark_predictions")
                torchvision.utils.save_image(mask.float(),f"{folder}/landmark_predictions/prueba_batch{index_batch}_{index}.png")
                #save dice score
                batch_test_dice_score += dice(z[index].int(), mask.int())



            test_dice_score = batch_test_dice_score / y_pred.shape[0]
                
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_dice_score= test_dice_score / len(dataloader)
    return test_loss, test_dice_score


# 1. Take in various parameters required for training and test steps
def train(
        device,
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
        epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)

        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,device=device, epoch = epoch)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc.to('cpu').item())
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc.to('cpu').item())

    # 6. Return the filled results at the end of the epochs
    return results



def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()