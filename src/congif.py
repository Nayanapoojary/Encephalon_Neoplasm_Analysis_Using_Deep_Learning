import os
from pathlib import Path
import torch

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "brisc2025"
    MODEL_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Create directories if they don't exist
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Classification task paths
    CLASSIFICATION_TRAIN = "C:\Users\navee\Downloads\encephalon_neoplasm_major_project\data\preprocessed\classification_task\train"
    CLASSIFICATION_TEST = "C:\Users\navee\Downloads\encephalon_neoplasm_major_project\data\preprocessed\classification_task\test"
    
    # Segmentation task paths
    SEGMENTATION_TRAIN_IMAGES = "C:\Users\navee\Downloads\encephalon_neoplasm_major_project\data\preprocessed\segmentation_task\train\images"
    SEGMENTATION_TRAIN_MASKS = "C:\Users\navee\Downloads\encephalon_neoplasm_major_project\data\preprocessed\segmentation_task\train\masks"
    SEGMENTATION_TEST_IMAGES = "C:\Users\navee\Downloads\encephalon_neoplasm_major_project\data\preprocessed\segmentation_task\test\images"
    SEGMENTATION_TEST_MASKS = "C:\Users\navee\Downloads\encephalon_neoplasm_major_project\data\preprocessed\segmentation_task\test\masks"
    
    # Model parameters
    CLASSIFICATION_CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    VIEW_CLASSES = ['axial', 'sagittal', 'coronal']  # top, side, front
    NUM_CLASSES = len(CLASSIFICATION_CLASSES)
    NUM_VIEW_CLASSES = len(VIEW_CLASSES)
    
    # Image parameters
    IMAGE_SIZE = 224
    SEGMENTATION_SIZE = 256
    BATCH_SIZE = 8  # Optimized for Intel i5
    
    # Training parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    PATIENCE = 10  # Early stopping
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = min(4, os.cpu_count())  # Optimized for Intel i5
    
    # Data augmentation parameters
    ROTATION_RANGE = 20
    BRIGHTNESS_RANGE = 0.2
    CONTRAST_RANGE = 0.2
    
    # Severity classification thresholds (based on tumor area percentage)
    SEVERITY_THRESHOLDS = {
        'mild': 0.05,      # < 5% of brain area
        'moderate': 0.15,  # 5-15% of brain area
        'severe': float('inf')  # > 15% of brain area
    }
    
    # Model save paths
    CLASSIFICATION_MODEL_PATH = MODEL_DIR / "tumor_classifier.pth"
    SEGMENTATION_MODEL_PATH = MODEL_DIR / "tumor_segmentation.pth"
    VIEW_CLASSIFIER_MODEL_PATH = MODEL_DIR / "view_classifier.pth"
    
    # Logging
    LOG_DIR = RESULTS_DIR / "logs"
    LOG_DIR.mkdir(exist_ok=True)