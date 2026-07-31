"""
Data loading utilities for Fake Currency Detection.

Expected folder layout (ImageFolder-compatible, single directory,
NO manual train/test moving needed - the split happens in code):

    data/
        Real/
            10/    <- denomination subfolders, read recursively
            20/
            50/
            100/
            200/
            500/
            2000/
        Fake/
            10/
            20/
            50/
            100/
            200/
            500/
            2000/

Class labels are assigned alphabetically by torchvision (Fake -> 0, Real -> 1).
You can verify this any time via `dataset.classes` / `dataset.class_to_idx`.

An 80/20 train/test split is created internally (stratified by class, so the
real/fake ratio is preserved in both splits) - you never need to move files
into separate folders yourself.
"""

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMG_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 2
TEST_SPLIT = 0.2  # 20% of data for testing, 80% for training
SEED = 42


def get_transforms(train: bool = True):
    """Return torchvision transforms. Light augmentation for training only."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])


def stratified_split_indices(targets, test_split: float, seed: int = SEED):
    """
    Splits sample indices into train/test while preserving class ratios
    (so both Real and Fake stay proportionally represented in both splits).
    """
    targets = np.array(targets)
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []

    for class_label in np.unique(targets):
        class_indices = np.where(targets == class_label)[0]
        rng.shuffle(class_indices)

        n = len(class_indices)
        n_test = int(n * test_split)

        test_idx.extend(class_indices[:n_test])
        train_idx.extend(class_indices[n_test:])

    return train_idx, test_idx


def get_dataloaders(data_dir: str = "data", batch_size: int = BATCH_SIZE,
                     test_split: float = TEST_SPLIT):
    """
    Builds train/test DataLoaders straight from data_dir (data/Real, data/Fake).
    No manual file-moving needed - the 80/20 split happens here in code.
    Returns: train_loader, test_loader, class_names
    """
    # Two ImageFolder instances over the SAME directory: one with training-time
    # augmentation, one without. We then assign disjoint indices to each split,
    # so training images get augmented but test images stay clean.
    train_transform_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(train=True))
    eval_transform_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(train=False))

    class_names = train_transform_dataset.classes  # e.g. ['Fake', 'Real']
    targets = [label for _, label in train_transform_dataset.samples]

    train_idx, test_idx = stratified_split_indices(targets, test_split)

    train_dataset = Subset(train_transform_dataset, train_idx)
    test_dataset = Subset(eval_transform_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Classes found: {class_names}")
    print(f"Total images: {len(train_transform_dataset)}")
    print(f"Train samples (80%): {len(train_dataset)} | Test samples (20%): {len(test_dataset)}")

    return train_loader, test_loader, class_names
