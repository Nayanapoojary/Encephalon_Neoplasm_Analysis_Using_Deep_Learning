#!/usr/bin/env python3
"""
Simple Brain Tumor Inference Script
Works directly with your trained model without config dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

class BrainTumorClassifier(nn.Module):
    """Same model architecture as training"""
    
    def __init__(self, num_classes=4, architecture='resnet18'):
        super(BrainTumorClassifier, self).__init__()
        
        if architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=False)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            self.backbone = models.resnet18(pretrained=False)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint"""
    
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get architecture
    architecture = checkpoint.get('architecture', 'resnet18')
    
    # Initialize model
    model = BrainTumorClassifier(num_classes=4, architecture=architecture)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully: {architecture}")
    
    return model

def get_transforms():
    """Get image transforms (same as training validation)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def predict_single_image(model, image_path, transform, device):
    """Predict single image"""
    
    class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()
    
    # Prepare result
    result = {
        'image_path': str(image_path),
        'image_name': Path(image_path).name,
        'predicted_class': class_names[predicted_class_idx],
        'predicted_class_idx': predicted_class_idx,
        'confidence': confidence,
        'glioma_prob': probabilities[0, 0].item(),
        'meningioma_prob': probabilities[0, 1].item(),
        'no_tumor_prob': probabilities[0, 2].item(),
        'pituitary_prob': probabilities[0, 3].item()
    }
    
    return result

def predict_directory(model, input_dir, transform, device, output_path):
    """Predict all images in directory"""
    
    print(f"Scanning directory: {input_dir}")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    
    input_path = Path(input_dir)
    for ext in image_extensions:
        image_paths.extend(input_path.glob(f"*{ext}"))
        image_paths.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Process all images
    results = []
    errors = []
    
    for image_path in tqdm(image_paths, desc="Predicting"):
        try:
            result = predict_single_image(model, image_path, transform, device)
            results.append(result)
        except Exception as e:
            errors.append(f"Error with {image_path}: {e}")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Print summary
        print(f"\nPREDICTION SUMMARY:")
        print(f"Total images: {len(results)}")
        print(f"Class distribution:")
        class_counts = df['predicted_class'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
        
        print(f"Average confidence: {df['confidence'].mean():.2%}")
        
        return results
    
    return []

def main():
    parser = argparse.ArgumentParser(description='Simple Brain Tumor Inference')
    parser.add_argument('--model-path', default='results/best_model.pth', help='Path to trained model')
    parser.add_argument('--input', required=True, help='Input image file or directory')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--device', default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        return
    
    # Load model
    device = torch.device(args.device)
    model = load_model(args.model_path, device)
    transform = get_transforms()
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single image prediction
        print(f"Predicting single image: {args.input}")
        result = predict_single_image(model, args.input, transform, device)
        
        print(f"\nPREDICTION RESULT:")
        print(f"Image: {result['image_name']}")
        print(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%} confidence)")
        print(f"All probabilities:")
        for cls in ['glioma', 'meningioma', 'no_tumor', 'pituitary']:
            prob = result[f"{cls}_prob"]
            print(f"  {cls}: {prob:.2%}")
        
        # Save single result as CSV
        df = pd.DataFrame([result])
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nResult saved to: {args.output}")
        
    elif os.path.isdir(args.input):
        # Directory prediction
        predict_directory(model, args.input, transform, device, args.output)
    
    else:
        print(f"Input path not found: {args.input}")

if __name__ == "__main__":
    main()