import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys
import numpy as np

# Add src to path - works from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if 'predict_result' in script_dir else script_dir
sys.path.append(os.path.join(project_root, 'src'))

from models.classification_model import BrainTumorClassifier

class TumorPredictor:
    def __init__(self, model_path='best_model.pth', device=None):
        """Initialize the tumor predictor"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Define image transformations (matching training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {self.device}")
    
    def _load_model(self, model_path):
        """Load the trained model - auto-detects architecture"""
        # Load checkpoint first to inspect architecture
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get state dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        
        # Detect architecture from state dict (ResNet18 vs ResNet50)
        is_resnet18 = 'backbone.layer1.0.conv1.weight' in state_dict and \
                      state_dict['backbone.layer1.0.conv1.weight'].shape[1] == 64
        
        # Create model with correct architecture
        import torchvision.models as models
        from torchvision.models import ResNet18_Weights
        
        if is_resnet18:
            print("Detected ResNet18 architecture")
            # Create ResNet18-based model
            model = nn.Module()
            model.backbone = models.resnet18(weights=None)
            num_features = model.backbone.fc.in_features
            
            # Check if model uses simple fc or Sequential fc
            has_sequential_fc = any('backbone.fc.1.weight' in k for k in state_dict.keys())
            
            if has_sequential_fc:
                print("Using Sequential FC layers")
                model.backbone.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(256, 4)
                )
            else:
                print("Using simple FC layer")
                model.backbone.fc = nn.Linear(num_features, 4)
            
            # Add attention if present in checkpoint
            if any('attention' in k for k in state_dict.keys()):
                from models.classification_model import SpatialAttention
                model.attention = SpatialAttention()
                model.forward = lambda x: model.backbone(model.attention(x) * x)
            else:
                model.forward = lambda x: model.backbone(x)
        else:
            print("Detected ResNet50 architecture")
            model = BrainTumorClassifier(num_classes=4, pretrained=False)
        
        # Load weights
        model.load_state_dict(state_dict)
        
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            print(f"Loaded from epoch {checkpoint['epoch']}")
            if 'best_val_acc' in checkpoint:
                print(f"Model accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        model.to(self.device)
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def validate_brain_mri(self, image_path):
        """
        Validate if image appears to be a brain MRI scan
        Returns (is_valid, validation_score, reasons)
        """
        from PIL import ImageStat
        
        image = Image.open(image_path).convert('RGB')
        
        validation_checks = []
        score = 0
        
        # Check 1: Grayscale-like (medical images are usually grayscale)
        r, g, b = image.split()
        r_std = np.std(np.array(r))
        g_std = np.std(np.array(g))
        b_std = np.std(np.array(b))
        
        # If R,G,B channels are very similar, it's likely grayscale
        channel_diff = abs(r_std - g_std) + abs(g_std - b_std) + abs(r_std - b_std)
        if channel_diff < 20:  # Channels are similar
            score += 1
            validation_checks.append("✓ Grayscale-like image")
        else:
            validation_checks.append("✗ Colorful image (MRIs are grayscale)")
        
        # Check 2: Dark background (MRIs typically have dark backgrounds)
        gray = image.convert('L')
        pixels = np.array(gray)
        dark_pixels = np.sum(pixels < 50) / pixels.size
        if dark_pixels > 0.3:  # >30% dark pixels
            score += 1
            validation_checks.append("✓ Has dark background")
        else:
            validation_checks.append("✗ No dark background (unlike typical MRI)")
        
        # Check 3: Intensity distribution (MRIs have specific patterns)
        mean_intensity = np.mean(pixels)
        if 40 < mean_intensity < 150:  # MRIs typically in this range
            score += 1
            validation_checks.append("✓ Intensity in MRI range")
        else:
            validation_checks.append(f"✗ Unusual intensity (mean: {mean_intensity:.1f})")
        
        # Check 4: Aspect ratio (brain MRIs are usually squarish)
        width, height = image.size
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio < 1.5:  # Relatively square
            score += 1
            validation_checks.append("✓ Appropriate aspect ratio")
        else:
            validation_checks.append(f"✗ Unusual aspect ratio ({aspect_ratio:.2f})")
        
        # Final decision: Need at least 3/4 checks to pass
        is_valid_mri = score >= 3
        
        return is_valid_mri, score, validation_checks
    
    def predict(self, image_path, return_all_probs=False):
        """
        Predict tumor type for a single image with comprehensive validation
        """
        # STEP 1: Validate image is a brain MRI
        is_valid_mri, validation_score, validation_checks = self.validate_brain_mri(image_path)
        
        if not is_valid_mri:
            return {
                'image_path': image_path,
                'has_tumor': False,
                'tumor_type': 'Invalid Image',
                'confidence': 0.0,
                'prediction': 'invalid',
                'validation_score': f"{validation_score}/4",
                'validation_checks': validation_checks,
                'warning': f'Image validation failed ({validation_score}/4 checks passed). This does not appear to be a brain MRI scan.'
            }
        
        # STEP 2: Run model prediction
        image_tensor = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
        
        # Out-of-distribution detection using entropy and confidence
        all_probs = probabilities[0].cpu().numpy()
        entropy = -np.sum(all_probs * np.log(all_probs + 1e-10))
        max_entropy = np.log(len(self.class_names))
        normalized_entropy = entropy / max_entropy
        
        # Validation thresholds
        CONFIDENCE_THRESHOLD = 0.85
        ENTROPY_THRESHOLD = 0.55
        
        # Get second highest probability
        sorted_probs = np.sort(all_probs)[::-1]
        second_highest_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
        
        # Determine if image is valid brain MRI
        is_valid_image = True
        warnings = []
        
        if confidence_score < CONFIDENCE_THRESHOLD:
            is_valid_image = False
            warnings.append(f'Low confidence ({confidence_score*100:.2f}%)')
        
        if normalized_entropy > ENTROPY_THRESHOLD:
            is_valid_image = False
            warnings.append(f'High uncertainty (entropy: {normalized_entropy:.2f})')
        
        if not is_valid_image:
            result = {
                'image_path': image_path,
                'has_tumor': False,
                'tumor_type': 'Invalid Image',
                'confidence': confidence_score,
                'prediction': 'invalid',
                'entropy': normalized_entropy,
                'warning': ' | '.join(warnings) + ' - Image may not be a brain MRI scan.'
            }
        else:
            has_tumor = predicted_class != 'no_tumor'
            
            result = {
                'image_path': image_path,
                'has_tumor': has_tumor,
                'tumor_type': predicted_class if has_tumor else 'No Tumor',
                'confidence': confidence_score,
                'prediction': predicted_class,
                'entropy': normalized_entropy
            }
        
        # Add all class probabilities if requested
        if return_all_probs:
            result['class_probabilities'] = {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, all_probs)
            }
        
        return result
    
    def predict_batch(self, image_paths, return_all_probs=False):
        """Predict tumor types for multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path, return_all_probs)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def print_result(self, result):
        """Print prediction result in a formatted way"""
        print("\n" + "="*60)
        print(f"Image: {os.path.basename(result['image_path'])}")
        print("="*60)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        if 'warning' in result and result['prediction'] == 'invalid':
            print(f"⚠️  INVALID IMAGE DETECTED")
            print(f"   {result['warning']}")
            
            if 'validation_checks' in result:
                print(f"\nValidation Results ({result.get('validation_score', 'N/A')}):")
                for check in result['validation_checks']:
                    print(f"   {check}")
            
            print(f"\n   This image does not appear to be a brain MRI scan.")
            print(f"   Please provide a valid brain MRI image.")
            
            if 'class_probabilities' in result:
                print("\n   (Model attempted classification anyway):")
                for class_name, prob in result['class_probabilities'].items():
                    print(f"   {class_name:15s}: {prob*100:6.2f}%")
            return
        
        if result['has_tumor']:
            print(f"TUMOR DETECTED")
            print(f"   tumor: yes Type: {result['tumor_type'].upper()}")
            print(f"   Confidence: {result['confidence']*100:.2f}%")
        else:
            print(f"NO TUMOR DETECTED")
            print(f"   tumor: no")
            print(f"   Confidence: {result['confidence']*100:.2f}%")
        
        # Print all class probabilities if available
        if 'class_probabilities' in result:
            print("\nClass Probabilities:")
            predicted_class = result['prediction']
            for class_name, prob in result['class_probabilities'].items():
                if class_name == predicted_class:
                    print(f"   {class_name:15s}: {prob*100:6.2f}%")
                else:
                    print(f"   {class_name:15s}:   0.00%")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Brain Tumor Detection and Classification')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to directory containing images')
    parser.add_argument('--model', type=str, default='results/best_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--show_probs', action='store_true',
                       help='Show probabilities for all classes')
    parser.add_argument('--output', type=str, help='Path to save results (CSV format)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = TumorPredictor(model_path=args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure:")
        print("1. The model file exists at the specified path")
        print("2. The model was trained with BrainTumorClassifier architecture")
        return
    
    # Get images to process
    image_paths = []
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        image_paths = [args.image]
    
    elif args.image_dir:
        if not os.path.exists(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            return
        
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for file in os.listdir(args.image_dir):
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(args.image_dir, file))
        
        if not image_paths:
            print(f"No valid images found in {args.image_dir}")
            return
    
    else:
        print("Error: Please provide either --image or --image_dir")
        return
    
    # Make predictions
    print(f"\nProcessing {len(image_paths)} image(s)...")
    results = predictor.predict_batch(image_paths, return_all_probs=args.show_probs)
    
    # Print results
    for result in results:
        predictor.print_result(result)
    
    # Save results if output file specified
    if args.output:
        import csv
        
        with open(args.output, 'w', newline='') as f:
            fieldnames = ['image_path', 'has_tumor', 'tumor_type', 'confidence', 'is_valid']
            if args.show_probs:
                fieldnames.extend(['glioma_prob', 'meningioma_prob', 'no_tumor_prob', 'pituitary_prob'])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                if 'error' not in result:
                    is_valid = result['prediction'] != 'invalid'
                    row = {
                        'image_path': result['image_path'],
                        'has_tumor': result['has_tumor'],
                        'tumor_type': result['tumor_type'],
                        'confidence': f"{result['confidence']:.4f}",
                        'is_valid': is_valid
                    }
                    
                    if args.show_probs and 'class_probabilities' in result:
                        row['glioma_prob'] = f"{result['class_probabilities']['glioma']:.4f}"
                        row['meningioma_prob'] = f"{result['class_probabilities']['meningioma']:.4f}"
                        row['no_tumor_prob'] = f"{result['class_probabilities']['no_tumor']:.4f}"
                        row['pituitary_prob'] = f"{result['class_probabilities']['pituitary']:.4f}"
                    
                    writer.writerow(row)
        
        print(f"\nResults saved to {args.output}")
    
    # Summary
    valid_results = [r for r in results if r.get('prediction') != 'invalid']
    invalid_results = [r for r in results if r.get('prediction') == 'invalid']
    tumor_count = sum(1 for r in valid_results if r.get('has_tumor', False))
    no_tumor_count = len(valid_results) - tumor_count
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(results)}")
    print(f"Valid brain MRI scans: {len(valid_results)}")
    print(f"Invalid/Non-MRI images: {len(invalid_results)}")
    print(f"Tumor detected: {tumor_count}")
    print(f"No tumor: {no_tumor_count}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()