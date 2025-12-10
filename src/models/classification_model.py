import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
import torch.nn.functional as F

class BrainTumorClassifier(nn.Module):
    """
    Brain tumor classification model using pre-trained ResNet50
    Classifies into: glioma, meningioma, no_tumor, pituitary
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        super(BrainTumorClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Add attention mechanism
        self.attention = SpatialAttention()
        
    def forward(self, x):
        # Apply attention
        x = self.attention(x) * x
        
        # Forward through backbone
        features = self.backbone(x)
        
        return features

class EfficientNetClassifier(nn.Module):
    """
    Alternative classifier using EfficientNet-B0 (lighter for Intel i5)
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        
        if pretrained:
            self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class SpatialAttention(nn.Module):
    """Spatial attention mechanism to focus on important regions"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class SeverityClassifier(nn.Module):
    """
    Additional classifier to determine tumor severity
    Based on tumor characteristics and size
    """
    
    def __init__(self, input_features=2048, num_severity_classes=3):
        super(SeverityClassifier, self).__init__()
        
        self.severity_head = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_severity_classes)  # mild, moderate, severe
        )
    
    def forward(self, features):
        return self.severity_head(features)

class MultiTaskBrainTumorModel(nn.Module):
    """
    Multi-task model that performs both classification and severity assessment
    """
    
    def __init__(self, num_classes=4, num_severity_classes=3, pretrained=True):
        super(MultiTaskBrainTumorModel, self).__init__()
        
        # Shared backbone
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.features(dummy_input)
            feature_dim = features.view(features.size(0), -1).size(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Severity classification head
        self.severity_classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_severity_classes)
        )
        
        # Attention mechanism
        self.attention = SpatialAttention()
    
    def forward(self, x, return_features=False):
        # Apply attention
        x = self.attention(x) * x
        
        # Extract features
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        
        # Classification
        tumor_classification = self.classifier(features_flat)
        
        # Severity assessment
        severity_classification = self.severity_classifier(features_flat)
        
        if return_features:
            return tumor_classification, severity_classification, features_flat
        else:
            return tumor_classification, severity_classification

def create_classification_model(model_type='resnet50', num_classes=4, pretrained=True):
    """
    Factory function to create different types of classification models
    
    Args:
        model_type (str): Type of model ('resnet50', 'efficientnet', 'multitask')
        num_classes (int): Number of tumor classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: The created model
    """
    
    if model_type == 'resnet50':
        return BrainTumorClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'efficientnet':
        return EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'multitask':
        return MultiTaskBrainTumorModel(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different models
    models_to_test = ['resnet50', 'efficientnet', 'multitask']
    
    for model_type in models_to_test:
        print(f"\nTesting {model_type} model:")
        model = create_classification_model(model_type=model_type)
        model.to(device)
        
        # Count parameters
        param_count = count_parameters(model)
        print(f"Trainable parameters: {param_count:,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.no_grad():
            if model_type == 'multitask':
                tumor_out, severity_out = model(dummy_input)
                print(f"Tumor classification output shape: {tumor_out.shape}")
                print(f"Severity classification output shape: {severity_out.shape}")
            else:
                output = model(dummy_input)
                print(f"Output shape: {output.shape}")
        
        print("✓ Model test passed!")