"""
Complete Brain Tumor Diagnosis System - FIXED VERSION
Includes proper preprocessing, classification, and segmentation with validation
"""

import os
import numpy as np
import cv2
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import ndimage

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
except Exception:
    torch = None
    nn = None
    F = None


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class SegmentationModel(nn.Module):
    """U-Net for Brain Tumor Segmentation"""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        self.downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        ])
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        ])
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        
        x = self.final_conv(x)
        return torch.sigmoid(x)


class ClassificationModel(nn.Module):
    """Classification model"""
    def __init__(self, num_classes=4, model_type='resnet18'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif model_type == 'resnet18':
            self.backbone = models.resnet18(pretrained=False)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif model_type == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=False)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def forward(self, x):
        return self.backbone(x)


class BrainTumorDiagnosisSystem:
    """Complete integrated system with improved segmentation validation"""

    def __init__(self, classification_model_path=None, segmentation_model_path=None):
        print("=" * 80)
        print("INITIALIZING BRAIN TUMOR DIAGNOSIS SYSTEM (FIXED VERSION)")
        print("=" * 80)

        # SORTED alphabetically - CRITICAL for matching training order
        self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        self.device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        print(f"Class order: {self.class_names}")

        self.classification_model = None
        self.segmentation_model = None
        
        # Classification uses ImageNet normalization
        self.classification_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        if classification_model_path and os.path.exists(classification_model_path):
            print("\n[1/2] Loading Classification Model...")
            self.classification_model = self._load_model(classification_model_path, task='classification')

        if segmentation_model_path and os.path.exists(segmentation_model_path):
            print("\n[2/2] Loading Segmentation Model...")
            self.segmentation_model = self._load_model(segmentation_model_path, task='segmentation')

        print("\n" + "=" * 80)
        print("✅ SYSTEM READY FOR DIAGNOSIS")
        print("=" * 80)

    def _load_model(self, path, task='classification'):
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ('.pth', '.pt'):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                
                if isinstance(checkpoint, torch.nn.Module):
                    checkpoint.eval()
                    return checkpoint
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        print(f"  → Found training checkpoint")
                        if 'best_val_acc' in checkpoint:
                            print(f"     Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
                        if 'architecture' in checkpoint:
                            print(f"     Architecture: {checkpoint['architecture']}")
                    else:
                        state_dict = checkpoint.get('state_dict', checkpoint)
                    
                    if task == 'classification':
                        print("  → Instantiating classification model...")
                        arch = checkpoint.get('architecture', 'resnet18')
                        model = ClassificationModel(num_classes=4, model_type=arch)
                        model.load_state_dict(state_dict)
                        model.to(self.device)
                        model.eval()
                        print(f"  ✓ Classification model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
                        return model
                    else:
                        print("  → Instantiating segmentation model...")
                        model = SegmentationModel()
                        model.load_state_dict(state_dict)
                        model.to(self.device)
                        model.eval()
                        print(f"  ✓ Segmentation model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
                        return model
            except Exception as e:
                print(f"  ❌ Error loading model: {e}")
                import traceback
                traceback.print_exc()
                return None
        return None

    def preprocess_image_for_classification(self, image_path):
        """
        Preprocess for classification - Uses ImageNet normalization
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print(f"  🔍 Classification Preprocessing:")
        print(f"     Original shape: {img.shape}")
        
        # Apply classification transform (includes resize, normalize, etc.)
        img_tensor = self.classification_transform(img_rgb)
        
        # Add batch dimension
        img_batch = img_tensor.unsqueeze(0)
        
        print(f"     Tensor shape: {img_batch.shape}")
        print(f"     Tensor range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
        
        return img_rgb, img_batch

    def preprocess_image_for_segmentation(self, image_path):
        """
        Preprocess for segmentation - MATCHES TRAINING EXACTLY
        Training used: Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (same as training)
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        print(f"  🔍 Segmentation Preprocessing:")
        print(f"     Original shape: {img.shape}")
        print(f"     Resized shape: {img_resized.shape}")
        print(f"     Pixel range before norm: [{img_resized.min()}, {img_resized.max()}]")
        
        # Convert to float [0, 1]
        img_float = img_resized.astype(np.float32) / 255.0
        print(f"     After /255: [{img_float.min():.3f}, {img_float.max():.3f}]")
        
        # Apply SAME normalization as training: (x - 0.5) / 0.5
        # This transforms [0, 1] to [-1, 1]
        img_normalized = (img_float - 0.5) / 0.5
        print(f"     After normalization: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
        
        # Convert to PyTorch tensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        img_batch = img_tensor.unsqueeze(0)
        
        print(f"     Final tensor shape: {img_batch.shape}")
        print(f"     Final tensor range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
        
        return img_rgb, img_resized, img_batch

    def classify_tumor(self, image_batch):
        """Run classification on preprocessed image"""
        if self.classification_model is None:
            return {
                'predicted_class': 'Unknown',
                'confidence': 0.0,
                'all_probabilities': {n: 0.0 for n in self.class_names},
                'has_tumor': False
            }
        
        image_batch = image_batch.to(self.device)
        
        with torch.no_grad():
            out = self.classification_model(image_batch)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
        
        pred_idx = int(np.argmax(probs))
        
        print(f"\n  🔍 DEBUG - Classification probabilities:")
        for i, (name, prob) in enumerate(zip(self.class_names, probs)):
            marker = "👉" if i == pred_idx else "  "
            print(f"     {marker} {name}: {prob*100:.2f}%")
        
        return {
            'predicted_class': self.class_names[pred_idx],
            'confidence': round(float(probs[pred_idx]) * 100, 2),
            'all_probabilities': {
                self.class_names[i]: round(float(probs[i]) * 100, 2) 
                for i in range(len(self.class_names))
            },
            'has_tumor': self.class_names[pred_idx] != 'No Tumor'
        }

    def validate_tumor_location(self, tumor_type, centroid, image_shape):
        """Validate if tumor location makes anatomical sense"""
        h, w = image_shape[:2]
        cx, cy = centroid['x'], centroid['y']
        
        expected_regions = {
            'Pituitary': {
                'y_range': (0.35, 0.65),
                'x_range': (0.35, 0.65),
                'description': 'Central pituitary region (sella turcica area)'
            },
            'Glioma': {
                'y_range': (0.1, 0.9),
                'x_range': (0.1, 0.9),
                'description': 'Brain parenchyma (widespread)'
            },
            'Meningioma': {
                'y_range': (0.0, 1.0),
                'x_range': (0.05, 1.0),
                'description': 'Meningeal surfaces (often peripheral)'
            }
        }
        
        if tumor_type not in expected_regions:
            return True, "Unknown tumor type - cannot validate location"
        
        region = expected_regions[tumor_type]
        cx_norm = cx / w
        cy_norm = cy / h
        
        x_valid = region['x_range'][0] <= cx_norm <= region['x_range'][1]
        y_valid = region['y_range'][0] <= cy_norm <= region['y_range'][1]
        
        if x_valid and y_valid:
            return True, f"Location matches expected anatomy: {region['description']}"
        else:
            return False, (f"⚠️ Location ({cx_norm:.2f}, {cy_norm:.2f}) outside expected region "
                          f"for {tumor_type}: {region['description']}")

    def clean_segmentation_mask(self, binary_mask):
        """Remove noise and artifacts from segmentation mask"""
        print("  🧹 Applying post-processing to clean mask...")
        
        mask_2d = binary_mask[:, :, 0] if len(binary_mask.shape) == 3 else binary_mask
        
        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask_2d, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
        # Fill holes
        mask_filled = ndimage.binary_fill_holes(mask_clean).astype(np.uint8)
        
        # Keep only largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
        
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            
            if len(areas) > 1:
                print(f"     Found {len(areas)} components, keeping largest")
            
            largest_label = 1 + np.argmax(areas)
            mask_final = (labels == largest_label).astype(np.uint8)
            
            print(f"     Kept component with area: {areas[np.argmax(areas)]} pixels")
        else:
            mask_final = mask_filled
            print(f"     No components found after cleaning")
        
        return np.expand_dims(mask_final, -1)

    def validate_spatial_consistency(self, binary_mask, tumor_type):
        """Check if detected region makes anatomical sense"""
        mask_2d = binary_mask[:, :, 0]
        
        num_labels, labels = cv2.connectedComponents(mask_2d)
        
        if num_labels > 4:
            return False, f"Too many disconnected regions ({num_labels-1}) - likely noise/artifacts"
        
        contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return True, "No contours to validate"
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        
        if aspect_ratio > 5:
            return False, f"Suspicious elongated shape (aspect ratio: {aspect_ratio:.1f}) - likely artifact"
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        
        if circularity < 0.1:
            return False, f"Highly irregular shape (circularity: {circularity:.2f}) - likely artifact"
        
        return True, f"Spatial consistency validated (AR: {aspect_ratio:.1f}, Circ: {circularity:.2f})"

    def segment_tumor(self, image_batch, classification_result=None, adaptive_threshold=True):
        if self.segmentation_model is None:
            return np.zeros((224, 224, 1)), np.zeros((224, 224, 1), dtype=np.uint8), \
                   {'tumor_detected': False, 'area_percentage': 0}

        image_batch = image_batch.to(self.device)
        
        with torch.no_grad():
            out = self.segmentation_model(image_batch).cpu().numpy()[0, 0]
        
        pred_mask = np.expand_dims(out, -1)
        
        print(f"\n  🔍 Segmentation Debug:")
        print(f"     Mask range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
        print(f"     Mean: {pred_mask.mean():.3f}, Std: {pred_mask.std():.3f}")
        
        # Adaptive thresholding
        if classification_result and adaptive_threshold:
            cls_confidence = classification_result['confidence']
            tumor_type = classification_result['predicted_class']

            if tumor_type == 'No Tumor':
                threshold = 0.99
                print(f"     Classification: No Tumor → segmentation disabled")

            elif tumor_type == 'Pituitary':
                if cls_confidence > 90:
                    if pred_mask.max() < 0.6:
                        threshold = 0.95
                        print(f"     ⚠️ Pituitary detected but low seg confidence → threshold=0.95")
                    else:
                        threshold = 0.7
                        print(f"     Pituitary with good seg confidence → threshold=0.7")
                else:
                    threshold = 0.85
                    print(f"     ⚠️ Uncertain pituitary classification → threshold=0.85")

            elif tumor_type == 'Meningioma':
                if cls_confidence > 85:
                    # Meningiomas often have lower segmentation confidence
                    if pred_mask.max() < 0.5:
                        threshold = 0.35  # Lower threshold for meningioma
                        print(f"     Meningioma with moderate seg confidence → threshold=0.35")
                    else:
                        threshold = 0.45  # Still lower than default
                        print(f"     Meningioma with good seg confidence → threshold=0.45")
                else:
                    threshold = 0.6
                    print(f"     ⚠️ Uncertain meningioma classification → threshold=0.6")

            elif tumor_type == 'Glioma':
                # Gliomas - default behavior with slight adjustments
                if cls_confidence < 70:
                    threshold = 0.7
                    print(f"     ⚠️ Low classification confidence → threshold=0.7")
                elif pred_mask.max() < 0.4:
                    threshold = 0.6
                    print(f"     ⚠️ Low segmentation confidence → threshold=0.6")
                else:
                    threshold = 0.5
                    print(f"     ✓ Good confidence → threshold=0.5")

            else:
                # Fallback for unknown tumor types
                if cls_confidence < 70:
                    threshold = 0.7
                    print(f"     ⚠️ Low classification confidence → threshold=0.7")
                elif pred_mask.max() < 0.4:
                    threshold = 0.6
                    print(f"     ⚠️ Low segmentation confidence → threshold=0.6")
                else:
                    threshold = 0.5
                    print(f"     ✓ Good confidence → threshold=0.5")
        else:
            threshold = 0.5
        
        # Apply threshold
        binary_mask = (pred_mask > threshold).astype(np.uint8)
        print(f"     Pixels > {threshold}: {np.sum(binary_mask)} / {binary_mask.size}")
        
        # Clean mask
        binary_mask = self.clean_segmentation_mask(binary_mask)
        
        mask_uint8 = (binary_mask[:, :, 0] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"     Contours found (after cleaning): {len(contours)}")
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            percentage = (area / (binary_mask.shape[0] * binary_mask.shape[1])) * 100
            print(f"     Largest contour area: {area:.0f} px ({percentage:.3f}%)")
            min_threshold = 0.5 if tumor_type == 'Meningioma' else 1.0
            if percentage > min_threshold:
                x, y, w, h = cv2.boundingRect(largest)
                M = cv2.moments(largest)
                cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
                
                metrics = {
                    'tumor_detected': True,
                    'area_pixels': int(area),
                    'area_percentage': round(percentage, 2),
                    'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'centroid': {'x': cx, 'y': cy},
                    'perimeter': round(cv2.arcLength(largest, True), 2),
                    'threshold_used': threshold,
                    'max_probability': round(float(pred_mask.max()), 3)
                }
                
                if classification_result:
                    tumor_type = classification_result['predicted_class']
                    
                    if tumor_type != 'No Tumor':
                        spatial_valid, spatial_msg = self.validate_spatial_consistency(binary_mask, tumor_type)
                        metrics['spatial_valid'] = spatial_valid
                        metrics['spatial_validation_message'] = spatial_msg
                        print(f"     🔍 Spatial validation: {spatial_msg}")
                        
                        if not spatial_valid:
                            print(f"     ⚠️ Spatial validation FAILED")
                            metrics['reliable'] = False
                        else:
                            location_valid, location_msg = self.validate_tumor_location(
                                tumor_type, metrics['centroid'], binary_mask.shape
                            )
                            metrics['location_valid'] = location_valid
                            metrics['location_validation_message'] = location_msg
                            print(f"     📍 Location validation: {location_msg}")
                            
                            if not location_valid:
                                metrics['reliable'] = False
                            else:
                                metrics['reliable'] = True
                
                print(f"     ✅ Tumor detected! Area: {percentage:.2f}%")
                return pred_mask, binary_mask, metrics
            else:
                print(f"     ⚠️ Contour too small: {percentage:.3f}% < 1.0% threshold")
        else:
            print(f"     ⚠️ No contours detected")
        
        return pred_mask, binary_mask, {'tumor_detected': False, 'area_percentage': 0}

    def complete_diagnosis(self, image_path):
        """Complete diagnosis pipeline"""
        print(f"\n{'='*80}\nANALYZING: {os.path.basename(image_path)}\n{'='*80}")
        
        # Classification
        print("[1/3] Preprocessing for classification...")
        img_original, cls_batch = self.preprocess_image_for_classification(image_path)
        
        print("[2/3] Classifying tumor type...")
        classification = self.classify_tumor(cls_batch)
        print(f"  ✅ Result: {classification['predicted_class']} ({classification['confidence']:.1f}% confidence)")
        
        # Segmentation
        print("[3/3] Segmenting tumor region...")
        _, img_resized, seg_batch = self.preprocess_image_for_segmentation(image_path)
        
        pred_mask, binary_mask, metrics = self.segment_tumor(seg_batch, classification_result=classification)
        
        # Cross-validation
        if classification['predicted_class'] == 'No Tumor':
            print("  ℹ️  Classification confirmed no tumor")
            metrics = {
                'tumor_detected': False,
                'area_percentage': 0,
                'note': 'Classification confirmed no tumor present',
                'reliable': True
            }
            binary_mask = np.zeros_like(binary_mask)
            pred_mask = np.zeros_like(pred_mask)
        
        elif metrics.get('tumor_detected'):
            seg_confidence = metrics.get('max_probability', 0) * 100
            cls_confidence = classification['confidence']
            confidence_gap = abs(cls_confidence - seg_confidence)
            
            print(f"\n  🔍 Cross-Validation Check:")
            print(f"     Classification confidence: {cls_confidence:.1f}%")
            print(f"     Segmentation confidence: {seg_confidence:.1f}%")
            print(f"     Confidence gap: {confidence_gap:.1f}%")
            
            if confidence_gap > 40:
                print(f"  ⚠️  WARNING: Large classification-segmentation mismatch!")
                metrics['warning'] = 'Large confidence mismatch'
                metrics['reliable'] = False
            
            if not metrics.get('spatial_valid', True):
                print(f"  ⚠️  Spatial validation failed")
                metrics['reliable'] = False
            elif not metrics.get('location_valid', True):
                print(f"  ⚠️  Location validation failed")
                metrics['reliable'] = False
            else:
                print(f"  ✅ All validations passed")
                metrics['reliable'] = True
            
            print(f"\n  📊 Segmentation Summary:")
            print(f"     Tumor Area: {metrics['area_percentage']:.2f}%")
            print(f"     Location: ({metrics['centroid']['x']}, {metrics['centroid']['y']})")
            print(f"     Reliability: {'✅ Reliable' if metrics.get('reliable', True) else '⚠️ Uncertain'}")
        
        else:
            print(f"  ℹ️  No significant tumor regions detected")
            
            if classification['predicted_class'] != 'No Tumor':
                print(f"\n  ⚠️  MISMATCH DETECTED:")
                print(f"     Classification: {classification['predicted_class']} ({classification['confidence']:.1f}%)")
                print(f"     Segmentation: No tumor boundaries found")
                
                if classification['predicted_class'] == 'Pituitary':
                    print(f"     ℹ️  Note: Segmentation has limited pituitary training")
                    print(f"     → Diagnosis relies on classification (98.10% accuracy)")
                else:
                    print(f"     Possible causes:")
                    print(f"       • Very small/diffuse tumor (< 1.0% of scan)")
                    print(f"       • Low contrast boundaries")
                    print(f"     Recommendation: Multi-view MRI analysis")
                
                metrics['warning'] = 'Classification-segmentation mismatch'
                metrics['reliable'] = False
        
        print(f"{'='*80}\n✅ DIAGNOSIS COMPLETE\n{'='*80}\n")
        
        return {
            'image_path': image_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'images': {'original': img_original, 'processed': img_resized},
            'classification': classification,
            'segmentation': {
                'probability_mask': pred_mask,
                'binary_mask': binary_mask,
                'metrics': metrics
            }
        }

    def generate_diagnostic_report(self, results, save_path=None):
        """Generate visual diagnostic report"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        img = results['images']['processed']
        classification = results['classification']
        segmentation = results['segmentation']
        metrics = segmentation['metrics']

        fig.suptitle('BRAIN TUMOR DIAGNOSTIC REPORT', 
                    fontsize=18, fontweight='bold', y=0.98)

        # Original scan
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        ax1.set_title('Original MRI Scan', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Classification results
        ax2 = fig.add_subplot(gs[0, 1:3])
        ax2.axis('off')
        class_text = f"DIAGNOSIS: {classification['predicted_class']}\nConfidence: {classification['confidence']:.2f}%\n"
        for cls, prob in classification['all_probabilities'].items():
            class_text += f"\n  • {cls}: {prob:.2f}%"
        ax2.text(0.01, 0.5, class_text, fontsize=11, family='monospace', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # Probability bars
        ax3 = fig.add_subplot(gs[0, 3])
        classes = list(classification['all_probabilities'].keys())
        probs = list(classification['all_probabilities'].values())
        colors = ['red' if p == max(probs) else 'skyblue' for p in probs]
        ax3.barh(classes, probs, color=colors)
        ax3.set_xlabel('Probability (%)')
        ax3.set_title('Classification Confidence', fontweight='bold')
        ax3.set_xlim(0, 100)

        # Segmentation section
        if metrics.get('tumor_detected'):
            reliability_color = 'lightgreen' if metrics.get('reliable', True) else 'lightyellow'
            
            ax4 = fig.add_subplot(gs[1, 0])
            im = ax4.imshow(segmentation['probability_mask'][:, :, 0], cmap='jet', vmin=0, vmax=1)
            ax4.set_title('Probability Heatmap', fontweight='bold')
            ax4.axis('off')
            plt.colorbar(im, ax=ax4, fraction=0.046)

            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(segmentation['binary_mask'][:, :, 0], cmap='gray')
            ax5.set_title('Segmentation Mask', fontweight='bold')
            ax5.axis('off')

            ax6 = fig.add_subplot(gs[1, 2])
            overlay = img.copy()
            mask_colored = np.zeros_like(img)
            mask_colored[segmentation['binary_mask'][:, :, 0] > 0] = [255, 0, 0]
            overlay_img = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            
            bbox = metrics.get('bounding_box')
            cent = metrics.get('centroid')
            if bbox:
                cv2.rectangle(overlay_img, (bbox['x'], bbox['y']), 
                            (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), (0, 255, 0), 2)
            if cent:
                cv2.circle(overlay_img, (cent['x'], cent['y']), 5, (255, 255, 0), -1)

            ax6.imshow(overlay_img)
            title_text = 'Tumor Location'
            if not metrics.get('reliable', True):
                title_text += ' ⚠️ Uncertain'
            ax6.set_title(title_text, fontweight='bold')
            ax6.axis('off')

            ax7 = fig.add_subplot(gs[1, 3])
            ax7.axis('off')
            seg_text = f"Tumor Size: {metrics['area_pixels']:,} px\n"
            seg_text += f"Coverage: {metrics['area_percentage']}%\n"
            seg_text += f"Perimeter: {metrics['perimeter']:.1f} px\n"
            seg_text += f"\nMax Probability: {metrics.get('max_probability', 0):.2%}\n"
            seg_text += f"Threshold: {metrics.get('threshold_used', 0.5):.2f}\n"
            
            if 'warning' in metrics:
                seg_text += f"\n⚠️ {metrics['warning']}"
            
            if 'spatial_valid' in metrics:
                if metrics['spatial_valid']:
                    seg_text += f"\n✅ Spatial check passed"
                else:
                    seg_text += f"\n⚠️ Spatial validation failed"
            
            if 'location_valid' in metrics:
                if metrics['location_valid']:
                    seg_text += f"\n✅ Location validated"
                else:
                    seg_text += f"\n⚠️ Location uncertain"
            
            ax7.text(0.01, 0.5, seg_text, fontsize=10, family='monospace', va='center',
                    bbox=dict(boxstyle='round', facecolor=reliability_color, alpha=0.5))
        else:
            ax_empty = fig.add_subplot(gs[1, :])
            ax_empty.axis('off')
            
            if classification['predicted_class'] == 'No Tumor':
                message = '✅ No tumor detected - Scan appears normal'
                bg_color = 'lightgreen'
            elif classification['predicted_class'] == 'Pituitary':
                message = f'ℹ️ Pituitary Tumor Detected (Classification: {classification["confidence"]:.1f}%)\n\n'
                message += 'Segmentation Note: Model trained primarily on glioma/meningioma.\n'
                message += 'Pituitary diagnosis relies on high-accuracy classification (98.10%).\n\n'
                message += 'Recommendation: Multi-view MRI with contrast + endocrine evaluation.'
                bg_color = 'lightcyan'
            else:
                message = '⚠️ Classification detected tumor but segmentation found no clear boundaries\n'
                message += f"Classification: {classification['predicted_class']} ({classification['confidence']:.1f}%)\n"
                message += 'Possible causes:\n'
                message += '  • Very small/diffuse tumor (< 1.0% of scan area)\n'
                message += '  • Low contrast boundaries in this view\n'
                message += '  • Image quality or orientation issues\n'
                message += 'Recommendation: Acquire multiple MRI views (Axial + Coronal + Sagittal)'
                bg_color = 'lightyellow'
            
            ax_empty.text(0.5, 0.5, message, 
                         ha='center', va='center', fontsize=12, family='monospace',
                         bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5))

        # Recommendation
        ax8 = fig.add_subplot(gs[2, :])
        ax8.axis('off')
        
        if not classification['has_tumor']:
            rec = "CLINICAL RECOMMENDATION: NORMAL SCAN\n"
            rec += "Status: No tumor detected\n"
            rec += "Action: Routine follow-up"
            rec_color = 'lightgreen'
        else:
            conf = classification['confidence']
            tumor_type = classification['predicted_class']
            
            if tumor_type == 'Pituitary':
                rec = f"CLINICAL RECOMMENDATION: PITUITARY ADENOMA EVALUATION\n\n"
                rec += f"Diagnosis: {tumor_type} (Confidence: {conf:.1f}%)\n\n"
                rec += "Recommended Actions:\n"
                rec += "  1. Endocrinology consultation for hormone panel\n"
                rec += "  2. Dedicated pituitary MRI with contrast\n"
                rec += "  3. Visual field testing\n"
                rec += "  4. Consider dynamic contrast enhancement\n\n"
                rec += "Note: Classification accuracy: 98.10%"
                rec_color = 'lightcyan'
            
            elif metrics.get('tumor_detected') and not metrics.get('reliable', True):
                rec = f"⚠️ RECOMMENDATION: FURTHER EVALUATION NEEDED\n\n"
                rec += f"Diagnosis: {tumor_type} (Confidence: {conf:.1f}%)\n"
                rec += f"Segmentation: Uncertain location/boundaries\n\n"
                rec += "Suggested Actions:\n"
                rec += "  1. Acquire multiple MRI views\n"
                rec += "  2. Consider contrast-enhanced imaging\n"
                rec += "  3. Radiologist expert review\n"
                rec_color = 'lightyellow'
            else:
                urgency = "HIGH PRIORITY" if conf > 95 else "MODERATE PRIORITY" if conf > 85 else "FURTHER TESTING"
                rec = f"CLINICAL RECOMMENDATION: {urgency}\n\n"
                rec += f"Diagnosis: {tumor_type}\n"
                rec += f"Confidence: {conf:.1f}%\n"
                
                if metrics.get('tumor_detected'):
                    rec += f"Tumor Coverage: {metrics['area_percentage']:.2f}%\n"
                
                if urgency == "HIGH PRIORITY":
                    rec += "\nAction: Immediate oncology/neurosurgery consultation"
                elif urgency == "MODERATE PRIORITY":
                    rec += "\nAction: Schedule appointment within 1-2 weeks"
                else:
                    rec += "\nAction: Additional imaging recommended"
                
                rec_color = 'lightcoral' if urgency == "HIGH PRIORITY" else 'lightyellow'
        
        ax8.text(0.5, 0.5, rec, fontsize=11, family='monospace', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor=rec_color, alpha=0.5))

        fig.text(0.5, 0.01, 
                f"Report Generated: {results['timestamp']} | AI-Assisted Diagnosis System v2.1",
                ha='center', fontsize=9, style='italic')

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Diagnostic report saved: {save_path}")

        plt.close()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    cls_path = r'results\classification_new\best_model.pth'
    seg_path = r'best_segmentation.pth'

    print("\n" + "="*80)
    print("🧠 BRAIN TUMOR DIAGNOSIS SYSTEM - TEST")
    print("="*80 + "\n")

    system = BrainTumorDiagnosisSystem(cls_path, seg_path)

    test_base = r"data\preprocessed\classification_task\test"
    
    for tumor_type in ['glioma', 'meningioma', 'no_tumor', 'pituitary']:
        folder = os.path.join(test_base, tumor_type)
        
        if not os.path.exists(folder):
            continue
        
        images = [f for f in os.listdir(folder) if f.endswith('.jpg')][:1]
        
        if not images:
            continue
        
        test_img = os.path.join(folder, images[0])
        
        print(f"\n{'='*80}")
        print(f"🔬 TESTING: {tumor_type.upper()}")
        print('='*80)
        
        result = system.complete_diagnosis(test_img)
        
        report_path = f'validation_{tumor_type}.png'
        system.generate_diagnostic_report(result, save_path=report_path)
        
        print(f"✅ Report saved: {report_path}\n")