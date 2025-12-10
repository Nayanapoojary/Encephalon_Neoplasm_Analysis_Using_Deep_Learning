"""
Configuration file for brain tumor diagnosis system
"""

import os
import json
from pathlib import Path

# ============================================================================
# MODEL PATHS
# ============================================================================
CLASSIFICATION_MODEL_PATH = 'results/classification_new/best_model.pth'
SEGMENTATION_MODEL_PATH = 'best_segmentation.pth'

# ============================================================================
# IMAGE SETTINGS
# ============================================================================
IMAGE_SIZE = (224, 224)
INPUT_CHANNELS = 3
NUM_CLASSES = 4

# ============================================================================
# CLASS NAMES (sorted alphabetically - CRITICAL for matching training!)
# ============================================================================
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# ============================================================================
# NORMALIZATION SETTINGS
# ============================================================================
# Classification uses ImageNet normalization
CLASSIFICATION_MEAN = [0.485, 0.456, 0.406]
CLASSIFICATION_STD = [0.229, 0.224, 0.225]

# Segmentation uses [-1, 1] normalization
SEGMENTATION_MEAN = [0.5, 0.5, 0.5]
SEGMENTATION_STD = [0.5, 0.5, 0.5]

# ============================================================================
# DEVICE SETTINGS
# ============================================================================
DEVICE = 'cpu'  # Will be auto-detected at runtime

# ============================================================================
# THRESHOLDS
# ============================================================================
SEGMENTATION_THRESHOLD = 0.5
MIN_TUMOR_AREA_PERCENTAGE = 1.0

# Tumor-specific thresholds
THRESHOLD_CONFIG = {
    'No Tumor': 0.99,
    'Pituitary': 0.7,
    'Meningioma': 0.45,
    'Glioma': 0.5
}

# ============================================================================
# FILE FORMATS
# ============================================================================
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.dcm', '.nii', '.nii.gz']

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================
MIN_IMAGE_SIZE = (50, 50)
MAX_IMAGE_SIZE = (4096, 4096)

# ============================================================================
# LOAD CONFIG FUNCTION
# ============================================================================

def load_config(config_path=None):
    """
    Load configuration from file or return default config
    
    Args:
        config_path (str, optional): Path to JSON config file
        
    Returns:
        dict: Configuration dictionary
    """
    
    # Default configuration
    config = {
        'model_paths': {
            'classification': CLASSIFICATION_MODEL_PATH,
            'segmentation': SEGMENTATION_MODEL_PATH
        },
        'image_settings': {
            'size': IMAGE_SIZE,
            'channels': INPUT_CHANNELS,
            'num_classes': NUM_CLASSES
        },
        'class_names': CLASS_NAMES,
        'normalization': {
            'classification': {
                'mean': CLASSIFICATION_MEAN,
                'std': CLASSIFICATION_STD
            },
            'segmentation': {
                'mean': SEGMENTATION_MEAN,
                'std': SEGMENTATION_STD
            }
        },
        'thresholds': {
            'segmentation': SEGMENTATION_THRESHOLD,
            'min_tumor_area': MIN_TUMOR_AREA_PERCENTAGE,
            'tumor_specific': THRESHOLD_CONFIG
        },
        'validation': {
            'min_size': MIN_IMAGE_SIZE,
            'max_size': MAX_IMAGE_SIZE,
            'supported_formats': SUPPORTED_IMAGE_FORMATS
        },
        'device': DEVICE
    }
    
    # If config file provided, load and merge
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Merge file config with defaults
            config.update(file_config)
            print(f"✅ Loaded config from: {config_path}")
        except Exception as e:
            print(f"⚠️ Error loading config file: {e}")
            print("Using default configuration")
    
    return config


def save_config(config, save_path='config.json'):
    """
    Save configuration to JSON file
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save config file
    """
    try:
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Config saved to: {save_path}")
    except Exception as e:
        print(f"❌ Error saving config: {e}")


def get_model_path(model_type='classification'):
    """
    Get model path for specified type
    
    Args:
        model_type (str): 'classification' or 'segmentation'
        
    Returns:
        str: Model file path
    """
    if model_type == 'classification':
        return CLASSIFICATION_MODEL_PATH
    elif model_type == 'segmentation':
        return SEGMENTATION_MODEL_PATH
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_threshold(tumor_type=None):
    """
    Get threshold for specified tumor type
    
    Args:
        tumor_type (str, optional): Tumor type name
        
    Returns:
        float: Threshold value
    """
    if tumor_type and tumor_type in THRESHOLD_CONFIG:
        return THRESHOLD_CONFIG[tumor_type]
    return SEGMENTATION_THRESHOLD


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'load_config',
    'save_config',
    'get_model_path',
    'get_threshold',
    'CLASS_NAMES',
    'IMAGE_SIZE',
    'CLASSIFICATION_MODEL_PATH',
    'SEGMENTATION_MODEL_PATH',
    'DEVICE'
]