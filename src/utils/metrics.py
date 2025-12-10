"""
Comprehensive metrics module for brain tumor classification and segmentation tasks.
Supports both classification and segmentation evaluation metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, classification_report
)
from typing import Dict, List, Tuple, Optional, Union
import warnings

class MetricsCalculator:
    """Handles calculation of various metrics for medical image analysis tasks."""
    
    def __init__(self, task_type: str = "classification", num_classes: int = None):
        """
        Initialize metrics calculator.
        
        Args:
            task_type: Either "classification" or "segmentation"
            num_classes: Number of classes for classification tasks
        """
        self.task_type = task_type.lower()
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: Optional[float] = None):
        """
        Update metrics with new batch of predictions and targets.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
            loss: Optional loss value for this batch
        """
        if self.task_type == "classification":
            # Convert to numpy and handle different tensor formats
            if isinstance(preds, torch.Tensor):
                if len(preds.shape) > 1 and preds.shape[1] > 1:
                    # Multi-class: take argmax
                    preds_np = torch.argmax(preds, dim=1).cpu().numpy()
                else:
                    # Binary: apply sigmoid and threshold
                    preds_np = (torch.sigmoid(preds).cpu().numpy() > 0.5).astype(int)
            else:
                preds_np = np.array(preds)
            
            if isinstance(targets, torch.Tensor):
                targets_np = targets.cpu().numpy()
            else:
                targets_np = np.array(targets)
            
            self.predictions.extend(preds_np.flatten())
            self.targets.extend(targets_np.flatten())
        
        elif self.task_type == "segmentation":
            # For segmentation, handle pixel-wise predictions
            if isinstance(preds, torch.Tensor):
                preds_np = torch.argmax(preds, dim=1).cpu().numpy()
            else:
                preds_np = np.array(preds)
            
            if isinstance(targets, torch.Tensor):
                targets_np = targets.cpu().numpy()
            else:
                targets_np = np.array(targets)
            
            self.predictions.extend(preds_np.flatten())
            self.targets.extend(targets_np.flatten())
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute_classification_metrics(self) -> Dict[str, float]:
        """Compute classification metrics."""
        if not self.predictions or not self.targets:
            return {}
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        accuracy = accuracy_score(targets, preds)
        
        # Handle binary vs multiclass
        average_mode = 'binary' if len(np.unique(targets)) == 2 else 'weighted'
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, preds, average=average_mode, zero_division=0
            )
        except Exception:
            precision = recall = f1 = 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_loss': float(np.mean(self.losses)) if self.losses else 0.0
        }
        
        # Add AUC for binary classification
        if len(np.unique(targets)) == 2:
            try:
                auc = roc_auc_score(targets, preds)
                metrics['auc'] = float(auc)
            except Exception:
                metrics['auc'] = 0.5
        
        return metrics
    
    def compute_segmentation_metrics(self) -> Dict[str, float]:
        """Compute segmentation metrics."""
        if not self.predictions or not self.targets:
            return {}
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Pixel accuracy
        accuracy = accuracy_score(targets, preds)
        
        # Dice coefficient (F1 score)
        try:
            _, _, f1, _ = precision_recall_fscore_support(
                targets, preds, average='weighted', zero_division=0
            )
        except Exception:
            f1 = 0.0
        
        # IoU (Intersection over Union)
        try:
            unique_classes = np.unique(np.concatenate([targets, preds]))
            ious = []
            for cls in unique_classes:
                pred_mask = (preds == cls)
                true_mask = (targets == cls)
                intersection = np.logical_and(pred_mask, true_mask).sum()
                union = np.logical_or(pred_mask, true_mask).sum()
                if union > 0:
                    ious.append(intersection / union)
            mean_iou = np.mean(ious) if ious else 0.0
        except Exception:
            mean_iou = 0.0
        
        metrics = {
            'pixel_accuracy': float(accuracy),
            'dice_coefficient': float(f1),
            'mean_iou': float(mean_iou),
            'avg_loss': float(np.mean(self.losses)) if self.losses else 0.0
        }
        
        return metrics
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute metrics based on task type."""
        if self.task_type == "classification":
            return self.compute_classification_metrics()
        elif self.task_type == "segmentation":
            return self.compute_segmentation_metrics()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def get_detailed_report(self) -> str:
        """Get detailed classification/segmentation report."""
        if not self.predictions or not self.targets:
            return "No data available for report"
        
        try:
            preds = np.array(self.predictions)
            targets = np.array(self.targets)
            
            if self.task_type == "classification":
                return classification_report(targets, preds, zero_division=0)
            else:
                # For segmentation, provide a simplified report
                unique_classes = np.unique(np.concatenate([targets, preds]))
                report = f"Segmentation Report - {len(unique_classes)} classes detected\n"
                report += f"Pixel Accuracy: {accuracy_score(targets, preds):.4f}\n"
                return report
        except Exception as e:
            return f"Error generating report: {str(e)}"


def setup_metrics(task_type: str, num_classes: int = None, device: str = 'cpu') -> MetricsCalculator:
    """
    Setup metrics calculator for the specified task.
    
    Args:
        task_type: Type of task ("classification" or "segmentation")
        num_classes: Number of classes (required for classification)
        device: Device to use for computations
    
    Returns:
        MetricsCalculator instance
    """
    if task_type.lower() not in ["classification", "segmentation"]:
        raise ValueError("task_type must be either 'classification' or 'segmentation'")
    
    if task_type.lower() == "classification" and num_classes is None:
        warnings.warn("num_classes not specified for classification task, defaulting to 2")
        num_classes = 2
    
    return MetricsCalculator(task_type=task_type, num_classes=num_classes)


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice score for segmentation.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice score
    """
    pred = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate IoU (Intersection over Union) for segmentation.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score
    """
    pred = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


# Convenience functions for common use cases
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy from raw logits."""
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Calculate top-k accuracy."""
    _, top_k_preds = torch.topk(logits, k, dim=1)
    targets_expanded = targets.unsqueeze(1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=1)
    return correct.float().mean().item()


# For backward compatibility
def get_metrics_for_task(task_type: str):
    """Get list of relevant metrics for a task type."""
    if task_type.lower() == "classification":
        return ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    elif task_type.lower() == "segmentation":
        return ['pixel_accuracy', 'dice_coefficient', 'mean_iou']
    else:
        return []


# Export main functions
__all__ = [
    'setup_metrics',
    'MetricsCalculator',
    'calculate_dice_score',
    'calculate_iou',
    'accuracy_from_logits',
    'top_k_accuracy',
    'get_metrics_for_task'
]