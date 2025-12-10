import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
import torch.nn.functional as F

class ViewClassifier(nn.Module):
    """
    Classifier to determine brain scan view orientation
    Classes: axial (top), sagittal (side), coronal (front)
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        super(ViewClassifier, self).__init__()
        
        # Use lighter ResNet18 for view classification (simpler task)
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class MobileViewClassifier(nn.Module):
    """
    Lightweight view classifier using MobileNetV2 (Intel i5 friendly)
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        super(MobileViewClassifier, self).__init__()
        
        if pretrained:
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ViewClassificationCNN(nn.Module):
    """
    Custom lightweight CNN for view classification
    """
    
    def __init__(self, num_classes=3):
        super(ViewClassificationCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class AdvancedViewClassifier(nn.Module):
    """
    Advanced view classifier with geometric features
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        super(AdvancedViewClassifier, self).__init__()
        
        # CNN backbone
        if pretrained:
            self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.cnn = models.resnet18(weights=None)
        
        # Remove final layer to get features
        self.features = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # Additional geometric feature extractor
        self.geometric_features = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 32)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 + 32, 256),  # 512 from ResNet18 + 32 from geometric
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # CNN features
        cnn_features = self.features(x)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # Geometric features
        geom_features = self.geometric_features(x)
        
        # Combine features
        combined_features = torch.cat([cnn_features, geom_features], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        
        return output

def analyze_image_geometry(image):
    """
    Analyze image geometric properties to help with view classification
    
    Args:
        image (torch.Tensor or numpy.ndarray): Input image
    
    Returns:
        dict: Dictionary containing geometric features
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:  # Batch dimension
            image = image[0]
        image = image.permute(1, 2, 0).cpu().numpy()
    
    import cv2
    import numpy as np
    
    # Convert to grayscale
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Basic properties
    height, width = gray.shape
    aspect_ratio = width / height
    
    # Find brain region (assuming brain is the brightest region)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find largest contour (brain region)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit ellipse to brain region
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            ellipse_aspect_ratio = ellipse[1][0] / ellipse[1][1]  # width/height of ellipse
            ellipse_angle = ellipse[2]
        else:
            ellipse_aspect_ratio = aspect_ratio
            ellipse_angle = 0
        
        # Calculate brain area
        brain_area = cv2.contourArea(largest_contour)
        brain_area_ratio = brain_area / (height * width)
    else:
        ellipse_aspect_ratio = aspect_ratio
        ellipse_angle = 0
        brain_area_ratio = 0.5
    
    # Calculate intensity distribution
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    return {
        'aspect_ratio': aspect_ratio,
        'ellipse_aspect_ratio': ellipse_aspect_ratio,
        'ellipse_angle': ellipse_angle,
        'brain_area_ratio': brain_area_ratio,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity
    }

def predict_view_from_geometry(geometric_features):
    """
    Simple rule-based view prediction from geometric features
    
    Args:
        geometric_features (dict): Dictionary of geometric features
    
    Returns:
        str: Predicted view ('axial', 'sagittal', 'coronal')
    """
    aspect_ratio = geometric_features['aspect_ratio']
    ellipse_aspect_ratio = geometric_features['ellipse_aspect_ratio']
    
    # Simple heuristics
    if aspect_ratio > 1.3 or ellipse_aspect_ratio > 1.3:
        return 'sagittal'  # Usually wider
    elif aspect_ratio < 0.7 or ellipse_aspect_ratio < 0.7:
        return 'coronal'   # Usually taller
    else:
        return 'axial'     # Usually more square

def create_view_classifier(model_type='resnet18', num_classes=3, pretrained=True):
    """
    Factory function to create different view classifiers
    
    Args:
        model_type (str): Type of model
        num_classes (int): Number of view classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: The created model
    """
    
    if model_type == 'resnet18':
        return ViewClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'mobilenet':
        return MobileViewClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'custom_cnn':
        return ViewClassificationCNN(num_classes=num_classes)
    elif model_type == 'advanced':
        return AdvancedViewClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == "__main__":
    # Test view classifiers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_to_test = ['resnet18', 'mobilenet', 'custom_cnn', 'advanced']
    
    for model_type in models_to_test:
        print(f"\nTesting {model_type} view classifier:")
        model = create_view_classifier(model_type=model_type)
        model.to(device)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {param_count:,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            
            # Test prediction
            probs = F.softmax(output, dim=1)
            predicted_classes = torch.argmax(probs, dim=1)
            print(f"Predicted classes: {predicted_classes}")
        
        print("✓ Model test passed!")
    
    # Test geometric analysis
    print("\nTesting geometric analysis:")
    dummy_image = torch.randn(1, 3, 224, 224)
    
    try:
        geometric_features = analyze_image_geometry(dummy_image)
        predicted_view = predict_view_from_geometry(geometric_features)
        print(f"Geometric features: {geometric_features}")
        print(f"Predicted view: {predicted_view}")
        print("✓ Geometric analysis test passed!")
    except ImportError:
        print("OpenCV not available for geometric analysis test")
    except Exception as e:
        print(f"Geometric analysis test failed: {e}")