import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import Config

class BrainTumorDataset(Dataset):
    """Dataset class for brain tumor classification"""
    
    def __init__(self, root_dir, transform=None, is_training=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_training = is_training
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(Config.CLASSIFICATION_CLASSES)}
        
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and labels"""
        for class_name in Config.CLASSIFICATION_CLASSES:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        self.samples.append({
                            'path': str(img_path),
                            'label': self.class_to_idx[class_name],
                            'class_name': class_name
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, sample['label']

class SegmentationDataset(Dataset):
    """Dataset class for brain tumor segmentation"""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_files = sorted([f for f in self.images_dir.glob("*") 
                                  if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding mask
        mask_path = self.masks_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Create empty mask if not found
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.long()

class ViewClassificationDataset(Dataset):
    """Dataset for classifying brain scan views (axial, sagittal, coronal)"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        self._create_view_labels()
    
    def _create_view_labels(self):
        """Create view labels based on image characteristics or filename patterns"""
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        # Simple heuristic: assign view based on image dimensions or filename
                        view_label = self._determine_view(img_path)
                        self.samples.append({
                            'path': str(img_path),
                            'view_label': view_label
                        })
    
    def _determine_view(self, img_path):
        """Determine view based on filename patterns or image analysis"""
        filename = img_path.name.lower()
        
        # Simple heuristic based on common naming patterns
        if any(term in filename for term in ['axial', 'ax', 'top']):
            return 0  # axial
        elif any(term in filename for term in ['sagittal', 'sag', 'side']):
            return 1  # sagittal
        elif any(term in filename for term in ['coronal', 'cor', 'front']):
            return 2  # coronal
        else:
            # Default classification based on image analysis
            return self._analyze_image_for_view(img_path)
    
    def _analyze_image_for_view(self, img_path):
        """Analyze image to determine most likely view"""
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 0  # Default to axial
        
        # Simple heuristic: analyze image shape and intensity patterns
        # This is a simplified approach - you might want to use more sophisticated methods
        height, width = image.shape
        aspect_ratio = width / height
        
        if aspect_ratio > 1.2:
            return 1  # sagittal (typically wider)
        elif aspect_ratio < 0.8:
            return 2  # coronal (typically taller)
        else:
            return 0  # axial (typically square-ish)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = cv2.imread(sample['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, sample['view_label']

def get_classification_transforms(is_training=True):
    """Get transforms for classification task"""
    if is_training:
        return A.Compose([
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            A.Rotate(limit=Config.ROTATION_RANGE, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=Config.BRIGHTNESS_RANGE,
                contrast_limit=Config.CONTRAST_RANGE,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def get_segmentation_transforms(is_training=True):
    """Get transforms for segmentation task"""
    if is_training:
        return A.Compose([
            A.Resize(Config.SEGMENTATION_SIZE, Config.SEGMENTATION_SIZE),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(Config.SEGMENTATION_SIZE, Config.SEGMENTATION_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def create_data_loaders():
    """Create all required data loaders"""
    
    # Classification data loaders
    train_transform = get_classification_transforms(is_training=True)
    val_transform = get_classification_transforms(is_training=False)
    
    train_dataset = BrainTumorDataset(
        Config.CLASSIFICATION_TRAIN, 
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = BrainTumorDataset(
        Config.CLASSIFICATION_TEST, 
        transform=val_transform,
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Segmentation data loaders
    seg_train_transform = get_segmentation_transforms(is_training=True)
    seg_val_transform = get_segmentation_transforms(is_training=False)
    
    seg_train_dataset = SegmentationDataset(
        Config.SEGMENTATION_TRAIN_IMAGES,
        Config.SEGMENTATION_TRAIN_MASKS,
        transform=seg_train_transform
    )
    
    seg_val_dataset = SegmentationDataset(
        Config.SEGMENTATION_TEST_IMAGES,
        Config.SEGMENTATION_TEST_MASKS,
        transform=seg_val_transform
    )
    
    seg_train_loader = DataLoader(
        seg_train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    seg_val_loader = DataLoader(
        seg_val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # View classification data loaders
    view_train_dataset = ViewClassificationDataset(
        Config.CLASSIFICATION_TRAIN,
        transform=train_transform
    )
    
    view_val_dataset = ViewClassificationDataset(
        Config.CLASSIFICATION_TEST,
        transform=val_transform
    )
    
    view_train_loader = DataLoader(
        view_train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    view_val_loader = DataLoader(
        view_val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return {
        'classification': (train_loader, val_loader),
        'segmentation': (seg_train_loader, seg_val_loader),
        'view_classification': (view_train_loader, view_val_loader)
    }

if __name__ == "__main__":
    # Test data loading
    data_loaders = create_data_loaders()
    
    print("Data loaders created successfully!")
    print(f"Classification train batches: {len(data_loaders['classification'][0])}")
    print(f"Classification val batches: {len(data_loaders['classification'][1])}")
    print(f"Segmentation train batches: {len(data_loaders['segmentation'][0])}")
    print(f"View classification train batches: {len(data_loaders['view_classification'][0])}")