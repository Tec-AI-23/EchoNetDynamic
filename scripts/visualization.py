import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


def show_image_from_path(path):
    plt.imshow(np.asarray(Image.open(path)))
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def show_n_channel_image_tensor(tensor: torch.Tensor):
    complete_image = tensor[0]
    for channel in tensor[1:]:
        complete_image += channel
    plt.imshow(complete_image)
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def show_frame_from_torch_tensor(tensor: torch.Tensor):
    desired_shape = (3, 112, 112)
    if tensor.shape != desired_shape:
        print(f"MAKE SURE THE SHAPE IS {desired_shape}")
        return
    image = tensor.numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def show_mask_from_torch_tensor(tensor: torch.Tensor):
    image = tensor.numpy()
    plt.imshow(image)
    plt.tight_layout()
    plt.axis("off")
    plt.show()
