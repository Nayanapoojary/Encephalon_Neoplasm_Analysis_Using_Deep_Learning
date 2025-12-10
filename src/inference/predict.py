#!/usr/bin/env python3
"""
Prediction script for BRIAC2025 competition
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.classification_model import create_classification_model
from utils.data_utils import get_classification_transforms
from utils.config import load_config

logger = logging.getLogger(__name__)

class Predictor:
    """Model predictor for inference"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = 'auto'):
        self.device = self._setup_device(device)
        
        # Load checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
        elif 'config' in self.checkpoint:
            self.config = self.checkpoint['config']
        else:
            raise ValueError("No configuration found. Please provide config file or use model with embedded config.")
        
        # Setup model
        self.model = self._setup_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup transforms
        self.transform = self._setup_transforms()
        
        # Get class names if available
        self.class_names = self._get_class_names()
        
        logger.info(f"Predictor initialized with model: {self.config['model']['name']}")
        logger.info(f"Number of classes: {self.config['model']['num_classes']}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup prediction device"""
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_model(self) -> torch.nn.Module:
        """Setup model for inference"""
        model_config = self.config['model']
        model = create_classification_model(
            name=model_config['name'],
            num_classes=model_config['num_classes'],
            pretrained=False,  # We're loading trained weights
            dropout=0.0  # No dropout during inference
        )
        
        # Handle DataParallel models
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        return model.to(self.device)
    
    def _setup_transforms(self):
        """Setup image transforms"""
        data_config = self.config['data']
        _, transform = get_classification_transforms(
            image_size=data_config.get('image_size', 224),
            normalize=data_config.get('normalize', True)
        )
        return transform
    
    def _get_class_names(self) -> List[str]:
        """Get class names if available"""
        if 'class_names' in self.config:
            return self.config['class_names']
        else:
            num_classes = self.config['model']['num_classes']
            return [f'Class_{i}' for i in range(num_classes)]
    
    def predict_image(self, image_path: str, return_probabilities: bool = False) -> dict:
        """
        Predict single image
        
        Args:
            image_path: Path to image file
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        result = {
            'image_path': image_path,
            'predicted_class': self.class_names[predicted_class_idx],
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence
        }
        
        if return_probabilities:
            class_probabilities = {}
            for i, class_name in enumerate(self.class_names):
                class_probabilities[class_name] = probabilities[0, i].item()
            result['class_probabilities'] = class_probabilities
        
        return result
    
    def predict_batch(self, image_paths: List[str], batch_size: int = 32, 
                     return_probabilities: bool = False) -> List[dict]:
        """
        Predict batch of images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Load and preprocess batch
            for image_path in batch_paths:
                try:
                    image = Image.open(image_path).convert('RGB')
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    # Add dummy tensor to maintain batch consistency
                    batch_tensors.append(torch.zeros(3, 224, 224))
            
            if not batch_tensors:
                continue
            
            batch_input = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_input)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)
            
            # Process batch results
            for j, image_path in enumerate(batch_paths):
                predicted_class_idx = predicted_classes[j].item()
                confidence = probabilities[j, predicted_class_idx].item()
                
                result = {
                    'image_path': image_path,
                    'predicted_class': self.class_names[predicted_class_idx],
                    'predicted_class_idx': predicted_class_idx,
                    'confidence': confidence
                }
                
                if return_probabilities:
                    class_probabilities = {}
                    for k, class_name in enumerate(self.class_names):
                        class_probabilities[class_name] = probabilities[j, k].item()
                    result['class_probabilities'] = class_probabilities
                
                results.append(result)
        
        return results
    
    def predict_directory(self, input_dir: str, output_path: str, 
                         batch_size: int = 32, return_probabilities: bool = False):
        """
        Predict all images in a directory and save results
        
        Args:
            input_dir: Input directory containing images
            output_path: Output CSV file path
            batch_size: Batch size for inference
            return_probabilities: Whether to return class probabilities
        """
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(input_dir).glob(f'**/*{ext}'))
            image_paths.extend(Path(input_dir).glob(f'**/*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            raise ValueError(f"No images found in directory: {input_dir}")
        
        logger.info(f"Found {len(image_paths)} images")
        
        # Predict all images
        results = self.predict_batch(image_paths, batch_size, return_probabilities)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")
        
        return results

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BRIAC2025 Prediction Script')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional if config is embedded in model)')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input image file or directory')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (CSV for batch prediction)')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    
    parser.add_argument('--probabilities', action='store_true',
                       help='Include class probabilities in output')
    
    parser.add_argument('--single-image', action='store_true',
                       help='Predict single image (outputs JSON)')
    
    return parser.parse_args()

def main():
    """Main prediction function"""
    setup_logging()
    args = parse_arguments()
    
    try:
        # Initialize predictor
        predictor = Predictor(
            model_path=args.model_path,
            config_path=args.config,
            device=args.device
        )
        
        if args.single_image:
            # Single image prediction
            if not os.path.isfile(args.input):
                raise ValueError(f"Input file not found: {args.input}")
            
            result = predictor.predict_image(
                args.input, 
                return_probabilities=args.probabilities
            )
            
            # Save result as JSON
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Single image prediction saved to: {args.output}")
            logger.info(f"Predicted class: {result['predicted_class']} (confidence: {result['confidence']:.4f})")
        
        else:
            # Batch prediction
            if not os.path.isdir(args.input):
                raise ValueError(f"Input directory not found: {args.input}")
            
            results = predictor.predict_directory(
                input_dir=args.input,
                output_path=args.output,
                batch_size=args.batch_size,
                return_probabilities=args.probabilities
            )
            
            # Print summary statistics
            df = pd.DataFrame(results)
            class_counts = df['predicted_class'].value_counts()
            
            logger.info("Prediction Summary:")
            logger.info(f"Total images processed: {len(results)}")
            logger.info("Class distribution:")
            for class_name, count in class_counts.items():
                logger.info(f"  {class_name}: {count} images")
            
            logger.info(f"Average confidence: {df['confidence'].mean():.4f}")
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()