import os
from dataset import EchoDatasetMasks, EchoDatasetHeatmaps
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def get_loaders_masks(
    images_dir,
    masks_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    test_size=0.2,
    seed=42,
    pin_memory=True,
):
    image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
    mask_paths = [os.path.join(masks_dir, mask) for mask in os.listdir(masks_dir)]

    (
        train_image_paths,
        val_image_paths,
        train_mask_paths,
        val_mask_paths,
    ) = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=seed
    )

    print("TRAIN PATHS LENGTHS: images, masks")
    print(f"{len(train_image_paths)}, {len(train_mask_paths)}")

    print("VALIDATION PATHS LENGTHS: images, masks")
    print(f"{len(val_image_paths)}, {len(val_mask_paths)}")

    train_dataset = EchoDatasetMasks(
        train_image_paths, train_mask_paths, transform=train_transform
    )

    val_dataset = EchoDatasetMasks(
        val_image_paths, val_mask_paths, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_loaders_heatmaps(
    images_dir,
    heatmaps_dir,
    masks_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    test_size=0.2,
    seed=42,
    pin_memory=True,
):
    image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir)][:32]
    mask_paths = [os.path.join(masks_dir, mask) for mask in os.listdir(masks_dir)][:32]
    heatmap_paths = [
        os.path.join(heatmaps_dir, heatmap) for heatmap in os.listdir(heatmaps_dir)
    ][:32]

    (
        train_image_paths,
        val_image_paths,
        train_heatmaps_paths,
        val_heatmaps_paths,
        train_mask_paths,
        val_mask_paths,
    ) = train_test_split(
        image_paths, heatmap_paths, mask_paths, test_size=test_size, random_state=seed
    )

    print("TRAIN PATHS LENGTHS: images, masks, heatmaps")
    print(
        f"{len(train_image_paths)}, {len(train_mask_paths)}, {len(train_heatmaps_paths)}"
    )

    print("VALIDATION PATHS LENGTHS: images, masks, heatmaps")
    print(f"{len(val_image_paths)}, {len(val_mask_paths)}, {len(val_heatmaps_paths)}")

    train_dataset = EchoDatasetHeatmaps(
        train_image_paths,
        train_heatmaps_paths,
        train_mask_paths,
        transform=train_transform,
    )

    val_dataset = EchoDatasetHeatmaps(
        val_image_paths, val_heatmaps_paths, val_mask_paths, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader
