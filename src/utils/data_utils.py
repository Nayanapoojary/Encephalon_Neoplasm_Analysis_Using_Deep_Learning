"""
Data utilities for BRIAC2025 project
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class ClassificationDataset(Dataset):
    """Dataset class for classification tasks"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, target_transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load data paths and labels
        self.samples = self._load_samples()
        self.classes = sorted(set(sample[1] for sample in self.samples))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        logger.info(f"Number of classes: {len(self.classes)}")
    
    def _load_samples(self) -> List[Tuple[str, str]]:
        """Load image paths and corresponding labels"""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_path in class_dir.glob('*.jpg'):
                    samples.append((str(img_path), class_name))
                for img_path in class_dir.glob('*.png'):
                    samples.append((str(img_path), class_name))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_name = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

class SegmentationDataset(Dataset):
    """Dataset class for segmentation tasks"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, target_transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load image and mask paths
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Tuple[str, str]]:
        """Load image and mask paths"""
        samples = []
        images_dir = self.data_dir / self.split / 'images'
        masks_dir = self.data_dir / self.split / 'masks'
        
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(f"Images or masks directory not found in {self.data_dir / self.split}")
        
        for img_path in images_dir.glob('*.jpg'):
            mask_path = masks_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                samples.append((str(img_path), str(mask_path)))
        
        for img_path in images_dir.glob('*.png'):
            mask_path = masks_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                samples.append((str(img_path), str(mask_path)))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale for mask
        
        if self.transform:
            # Apply same transform to both image and mask
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.target_transform(mask) if self.target_transform else transforms.ToTensor()(mask)
        
        return image, mask.long().squeeze(0)

def get_classification_transforms(image_size: int = 224, 
                                 augmentation: Optional[Dict[str, Any]] = None,
                                 normalize: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transforms for classification tasks
    
    Args:
        image_size: Target image size
        augmentation: Augmentation parameters
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Train and validation transforms
    """
    # Base transforms
    base_transforms = [transforms.Resize((image_size, image_size))]
    
    # Training transforms with augmentation
    train_transforms = base_transforms.copy()
    
    if augmentation:
        if 'horizontal_flip' in augmentation:
            train_transforms.append(transforms.RandomHorizontalFlip(p=augmentation['horizontal_flip']))
        
        if 'rotation' in augmentation:
            train_transforms.append(transforms.RandomRotation(augmentation['rotation']))
        
        if 'color_jitter' in augmentation:
            jitter_strength = augmentation['color_jitter']
            train_transforms.append(transforms.ColorJitter(
                brightness=jitter_strength,
                contrast=jitter_strength,
                saturation=jitter_strength,
                hue=jitter_strength/2
            ))
    
    train_transforms.append(transforms.ToTensor())
    
    # Validation transforms (no augmentation)
    val_transforms = base_transforms + [transforms.ToTensor()]
    
    # Add normalization
    if normalize:
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        train_transforms.append(normalize_transform)
        val_transforms.append(normalize_transform)
    
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)

def get_segmentation_transforms(image_size: int = 512,
                               augmentation: Optional[Dict[str, Any]] = None,
                               normalize: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transforms for segmentation tasks
    
    Args:
        image_size: Target image size
        augmentation: Augmentation parameters
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Train and validation transforms for images and masks
    """
    # Base transforms
    base_transforms = [transforms.Resize((image_size, image_size))]
    
    # Training transforms
    train_transforms = base_transforms.copy()
    
    if augmentation:
        if 'horizontal_flip' in augmentation:
            train_transforms.append(transforms.RandomHorizontalFlip(p=augmentation['horizontal_flip']))
        
        if 'vertical_flip' in augmentation:
            train_transforms.append(transforms.RandomVerticalFlip(p=augmentation['vertical_flip']))
        
        if 'rotation' in augmentation:
            train_transforms.append(transforms.RandomRotation(augmentation['rotation']))
    
    train_transforms.append(transforms.ToTensor())
    
    # Validation transforms
    val_transforms = base_transforms + [transforms.ToTensor()]
    
    # Add normalization for images only
    if normalize:
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        train_transforms.append(normalize_transform)
        val_transforms.append(normalize_transform)
    
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)

def setup_data_loaders(config: Dict[str, Any], task: str) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Setup data loaders for training, validation, and testing
    
    Args:
        config: Configuration dictionary
        task: Task type ('classification' or 'segmentation')
        
    Returns:
        Train, validation, and test data loaders
    """
    data_config = config['data']
    training_config = config['training']
    
    # Get transforms
    if task == 'classification':
        train_transform, val_transform = get_classification_transforms(
            image_size=data_config.get('image_size', 224),
            augmentation=data_config.get('augmentation'),
            normalize=data_config.get('normalize', True)
        )
        dataset_class = ClassificationDataset
    
    elif task == 'segmentation':
        train_transform, val_transform = get_segmentation_transforms(
            image_size=data_config.get('image_size', 512),
            augmentation=data_config.get('augmentation'),
            normalize=data_config.get('normalize', True)
        )
        dataset_class = SegmentationDataset
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create datasets
    data_dir = config['data_path']
    
    try:
        train_dataset = dataset_class(data_dir, split='train', transform=train_transform)
        val_dataset = dataset_class(data_dir, split='val', transform=val_transform)
        
        # Try to create test dataset
        try:
            test_dataset = dataset_class(data_dir, split='test', transform=val_transform)
        except FileNotFoundError:
            logger.warning("Test split not found, using None for test dataset")
            test_dataset = None
    
    except FileNotFoundError as e:
        logger.error(f"Failed to create datasets: {e}")
        raise
    
    # Create data loaders
    batch_size = training_config['batch_size']
    num_workers = config.get('num_workers', 4)
    pin_memory = config.get('pin_memory', True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader) if test_loader else 0}")
    
    return train_loader, val_loader, test_loader

def calculate_class_weights(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced datasets
    
    Args:
        dataset: Dataset to calculate weights for
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    class_counts = torch.zeros(num_classes)
    
    for _, label in dataset:
        if isinstance(label, torch.Tensor):
            # For segmentation, count pixels per class
            unique, counts = torch.unique(label, return_counts=True)
            for cls, count in zip(unique, counts):
                if cls < num_classes:
                    class_counts[cls] += count
        else:
            # For classification
            class_counts[label] += 1
    
    # Calculate inverse frequency weights
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    
    # Handle zero counts
    class_weights[class_counts == 0] = 0
    
    logger.info(f"Calculated class weights: {class_weights}")
    
    return class_weights