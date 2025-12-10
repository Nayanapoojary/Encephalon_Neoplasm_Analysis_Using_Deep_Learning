"""
PyTorch Brain Tumor Segmentation Inference Script
Real-time prediction using PyTorch models
"""

import os
import glob
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import your PyTorch U-Net model
from unet_pytorch import UNet


class PyTorchSegmentationPredictor:
    """Real-time segmentation predictor using PyTorch"""
    
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Loading PyTorch segmentation model...")
        print(f"Device: {self.device}")
        
        # Load model
        self.model = UNet(in_channels=3, out_channels=1)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get model info
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"✅ Model loaded successfully!")
        print(f"Parameters: {total_params:,}")
        print(f"Best Dice Score: {checkpoint.get('best_dice', 'N/A')}")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for inference"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, target_size)
        
        # Normalize
        img_normalized = img_resized / 255.0
        
        # Convert to tensor (HWC to CHW)
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        return img_rgb, img_resized, img_tensor
    
    @torch.no_grad()
    def predict(self, image_path, threshold=0.5):
        """Predict tumor segmentation"""
        # Preprocess
        img_original, img_resized, img_tensor = self.preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)
        
        # Predict
        pred_mask = self.model(img_tensor)
        pred_mask = pred_mask.cpu().numpy()[0, 0]  # Remove batch and channel dims
        
        # Create binary mask
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        # Calculate metrics
        metrics = self.calculate_metrics(pred_binary)
        
        results = {
            'image_original': img_original,
            'image_resized': img_resized,
            'mask_probability': pred_mask,
            'mask_binary': pred_binary,
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_path': image_path
        }
        
        return results
    
    def calculate_metrics(self, binary_mask):
        """Calculate tumor metrics"""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {
                'tumor_detected': False,
                'tumor_area_pixels': 0,
                'tumor_area_percentage': 0.0,
                'bounding_box': None,
                'centroid': None,
                'confidence': 'N/A'
            }
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate area
        area = cv2.contourArea(largest_contour)
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        percentage = (area / total_pixels) * 100
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        return {
            'tumor_detected': True,
            'tumor_area_pixels': int(area),
            'tumor_area_percentage': round(percentage, 2),
            'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'centroid': {'x': int(cx), 'y': int(cy)},
            'perimeter': int(cv2.arcLength(largest_contour, True)),
            'confidence': round(np.mean(binary_mask) * 100, 2)
        }
    
    def visualize_results(self, results, save_path=None, show=True):
        """Visualize segmentation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        img = results['image_resized']
        mask_prob = results['mask_probability']
        mask_binary = results['mask_binary']
        metrics = results['metrics']
        
        # Row 1: Original, Probability, Binary
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original MRI Scan', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_prob, cmap='jet', vmin=0, vmax=1)
        axes[0, 1].set_title('Probability Heatmap', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)
        
        axes[0, 2].imshow(mask_binary, cmap='gray')
        axes[0, 2].set_title('Binary Mask', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Overlays and Analysis
        overlay = img.copy()
        mask_colored = np.zeros_like(img)
        mask_colored[mask_binary > 0] = [255, 0, 0]
        overlay_result = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[1, 0].imshow(overlay_result)
        axes[1, 0].set_title('Tumor Overlay', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Contour with bounding box
        contour_img = img.copy()
        if metrics['tumor_detected']:
            mask_uint8 = (mask_binary * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
            
            bbox = metrics['bounding_box']
            cv2.rectangle(contour_img, 
                         (bbox['x'], bbox['y']), 
                         (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                         (0, 255, 0), 2)
            
            centroid = metrics['centroid']
            cv2.circle(contour_img, (centroid['x'], centroid['y']), 5, (255, 255, 0), -1)
        
        axes[1, 1].imshow(contour_img)
        axes[1, 1].set_title('Contour & Bounding Box', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Metrics text
        axes[1, 2].axis('off')
        if metrics['tumor_detected']:
            bbox = metrics['bounding_box']
            metrics_text = f"""SEGMENTATION RESULTS

Status: TUMOR DETECTED ✓

Size:
• Area: {metrics['tumor_area_pixels']:,} px
• Coverage: {metrics['tumor_area_percentage']}%
• Perimeter: {metrics['perimeter']} px

Location:
• Center: ({metrics['centroid']['x']}, {metrics['centroid']['y']})
• Box: {bbox['width']}×{bbox['height']}

Confidence: {metrics['confidence']}%

Timestamp: {results['timestamp']}
            """
        else:
            metrics_text = f"""SEGMENTATION RESULTS

Status: NO TUMOR DETECTED

No significant tumor regions
detected in this scan.

Timestamp: {results['timestamp']}
            """
        
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle('Brain Tumor Segmentation Analysis (PyTorch)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def batch_predict(self, image_dir, output_dir=None, save_visualizations=True):
        """Predict on multiple images"""
        image_files = glob.glob(os.path.join(image_dir, '*.jpg')) + \
                      glob.glob(os.path.join(image_dir, '*.png'))
        
        print(f"\n{'='*80}")
        print(f"BATCH PREDICTION - Found {len(image_files)} images")
        print(f"{'='*80}\n")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        tumor_count = 0
        
        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing: {os.path.basename(img_path)}")
            
            try:
                results = self.predict(img_path)
                all_results.append(results)
                
                if results['metrics']['tumor_detected']:
                    tumor_count += 1
                    print(f"  ✓ Tumor detected - Area: {results['metrics']['tumor_area_percentage']}%")
                else:
                    print(f"  ✗ No tumor")
                
                if output_dir and save_visualizations:
                    vis_path = os.path.join(output_dir, f'result_{i}_{os.path.basename(img_path)}')
                    self.visualize_results(results, save_path=vis_path, show=False)
            
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Summary
        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        print(f"Total processed: {len(all_results)}")
        print(f"Tumors detected: {tumor_count}")
        print(f"No tumor: {len(all_results) - tumor_count}")
        print(f"Detection rate: {(tumor_count/len(all_results)*100):.1f}%")
        
        if output_dir:
            print(f"\nResults saved to: {output_dir}")
        
        return all_results


if __name__ == "__main__":
    print("="*80)
    print("PYTORCH BRAIN TUMOR SEGMENTATION INFERENCE")
    print("="*80)
    
    # Model path
    model_path = 'results/segmentation_pytorch_latest/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print("\n💡 Available models:")
        available_models = glob.glob('results/segmentation_pytorch_*/best_model.pth')
        if available_models:
            for model in available_models:
                print(f"  - {model}")
            model_path = available_models[0]
            print(f"\n✅ Using: {model_path}")
        else:
            print("  No PyTorch models found!")
            print("  Train a model first using: python train_segmentation_pytorch.py")
            exit(1)
    
    # Initialize predictor
    predictor = PyTorchSegmentationPredictor(model_path)
    
    # Find test images
    test_dir = 'data/preprocessed/segmentation_task/test/images'
    if not os.path.exists(test_dir):
        test_dir = 'data/test_samples'
    
    if os.path.exists(test_dir):
        print(f"\n📁 Test directory: {test_dir}\n")
        
        # Batch prediction
        predictor.batch_predict(
            image_dir=test_dir,
            output_dir='outputs/pytorch_segmentation_results',
            save_visualizations=True
        )
    else:
        print(f"\n⚠️ Test directory not found: {test_dir}")
        print("Please create test_dir and add MRI images")
    
    print("\n✅ Inference completed!")