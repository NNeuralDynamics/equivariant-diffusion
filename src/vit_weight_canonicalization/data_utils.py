import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple


def get_cifar10_dataloaders(batch_size: int = 128,
                           num_workers: int = 4,
                           val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-10 data loaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for validation/test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    # Split train into train/val
    train_size = int(len(full_train_dataset) * (1 - val_split))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Apply test transform to validation set
    val_dataset.dataset.transform = transform_test
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_imagenet_subset_dataloaders(batch_size: int = 64,
                                   num_workers: int = 4,
                                   subset_size: int = 10000):
    """
    Get a subset of ImageNet for faster experiments
    Note: Requires ImageNet to be downloaded separately
    """
    # This is a placeholder - would need actual ImageNet path
    pass