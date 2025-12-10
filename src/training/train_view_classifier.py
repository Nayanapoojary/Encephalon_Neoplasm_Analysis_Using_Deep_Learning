"""
Complete training script for brain tumor classification.
Supports multiple CNN architectures and comprehensive evaluation metrics.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.utils.metrics import setup_metrics
except ImportError:
    print("Warning: Could not import setup_metrics. Creating fallback...")
    # Fallback metrics class
    class FallbackMetrics:
        def __init__(self, task_type, num_classes=None):
            self.reset()
        def reset(self): 
            self.correct = 0
            self.total = 0
            self.losses = []
        def update(self, preds, targets, loss=None):
            if isinstance(preds, torch.Tensor) and len(preds.shape) > 1:
                preds = torch.argmax(preds, dim=1)
            self.correct += (preds == targets).sum().item()
            self.total += targets.size(0)
            if loss: self.losses.append(loss)
        def compute_metrics(self):
            acc = self.correct / self.total if self.total > 0 else 0
            avg_loss = np.mean(self.losses) if self.losses else 0
            return {'accuracy': acc, 'avg_loss': avg_loss}
    
    def setup_metrics(task_type, num_classes=None, device='cpu'):
        return FallbackMetrics(task_type, num_classes)


class BrainTumorDataset(Dataset):
    """Custom dataset for brain tumor images."""
    
    def __init__(self, data_dir: str, transform=None, class_to_idx=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        
        # Auto-detect classes from directory structure
        if class_to_idx is None:
            self.class_to_idx = {}
            class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
            self.classes = sorted([d.name for d in class_dirs])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())
        
        print(f"Detected classes: {self.classes}")
        print(f"Class mapping: {self.class_to_idx}")
        
        # Load all image paths and labels
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples")
        self._print_class_distribution()
    
    def _load_samples(self):
        """Load all image samples from the directory structure."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} does not exist")
                continue
            
            class_samples = []
            for ext in image_extensions:
                class_samples.extend(class_dir.glob(f'*{ext}'))
                class_samples.extend(class_dir.glob(f'*{ext.upper()}'))
            
            for img_path in class_samples:
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
            
            print(f"Class '{class_name}': {len(class_samples)} images")
    
    def _print_class_distribution(self):
        """Print the distribution of classes in the dataset."""
        class_counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        
        print("\nClass Distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load and convert image
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                image = torch.zeros(3, 224, 224)
            return image, label


class BrainTumorClassifier(nn.Module):
    """CNN classifier for brain tumor detection."""
    
    def __init__(self, num_classes: int, architecture: str = 'resnet18', pretrained: bool = True):
        super(BrainTumorClassifier, self).__init__()
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Load pretrained model
        if architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif architecture == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        elif architecture == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            self.backbone.classifier[6] = nn.Linear(self.backbone.classifier[6].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Add dropout for regularization
        if hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)


class Trainer:
    """Training manager for brain tumor classification."""
    
    def __init__(self, model, device, num_classes, class_names):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        
        # Initialize metrics
        self.train_metrics = setup_metrics('classification', num_classes, device)
        self.val_metrics = setup_metrics('classification', num_classes, device)
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0
        progress_bar = tqdm(dataloader, desc='Training')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            self.train_metrics.update(outputs, targets, loss.item())
            total_loss += loss.item()
            
            # Update progress bar
            current_metrics = self.train_metrics.compute_metrics()
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{current_metrics.get('accuracy', 0):.4f}"
            })
        
        return self.train_metrics.compute_metrics()
    
    def validate_epoch(self, dataloader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        self.val_metrics.reset()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Validation')
            
            for data, targets in progress_bar:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                # Store predictions for confusion matrix
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update metrics
                self.val_metrics.update(outputs, targets, loss.item())
                
                # Update progress bar
                current_metrics = self.val_metrics.compute_metrics()
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{current_metrics.get('accuracy', 0):.4f}"
                })
        
        metrics = self.val_metrics.compute_metrics()
        return metrics, all_preds, all_targets
    
    def train(self, train_loader, val_loader, epochs, optimizer, criterion, scheduler=None, save_dir=None):
        """Complete training loop."""
        best_val_acc = 0.0
        best_epoch = 0
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Model: {self.model.architecture}")
        print(f"Device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_metrics, val_preds, val_targets = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics['avg_loss'])
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
            
            # Update history
            self.history['train_loss'].append(train_metrics['avg_loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['avg_loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Print epoch results
            epoch_time = time.time() - start_time
            print(f"\nEpoch Results:")
            print(f"Train - Loss: {train_metrics['avg_loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val   - Loss: {val_metrics['avg_loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"Time: {epoch_time:.2f}s")
            
            if scheduler:
                print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch + 1
                
                if save_dir:
                    self.save_checkpoint(save_dir, epoch, optimizer, val_metrics['avg_loss'], is_best=True)
                
                print(f"★ New best validation accuracy: {best_val_acc:.4f}")
            
            # Save regular checkpoint every 10 epochs
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir, epoch, optimizer, val_metrics['avg_loss'])
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
        
        return self.history
    
    def save_checkpoint(self, save_dir, epoch, optimizer, loss, is_best=False):
        """Save model checkpoint."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'architecture': self.model.architecture,
            'history': self.history
        }
        
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, save_dir / filename)
        
        if is_best:
            print(f"Saved best model to {save_dir / filename}")


def get_transforms(input_size=224, augment=True):
    """Get data transforms for training and validation."""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', color='blue')
    ax2.plot(history['val_acc'], label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate plot
    if 'learning_rates' in history and history['learning_rates']:
        ax3.plot(history['learning_rates'], color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Learning Rate Schedule')
    
    # Best metrics summary
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    min_val_loss = min(history['val_loss'])
    
    metrics_text = f"""
    Best Validation Accuracy: {best_val_acc:.4f}
    Best Epoch: {best_epoch}
    Minimum Validation Loss: {min_val_loss:.4f}
    Final Train Accuracy: {history['train_acc'][-1]:.4f}
    Final Val Accuracy: {history['val_acc'][-1]:.4f}
    """
    
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax4.set_title('Training Summary')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def create_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Create and plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classifier')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--architecture', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet50', 'efficientnet_b0', 'vgg16'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--augment', action='store_true', default=True, help='Use data augmentation')
    parser.add_argument('--input-size', type=int, default=224, help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Prepare data transforms
    train_transform, val_transform = get_transforms(args.input_size, args.augment)
    
    # Create datasets
    print("Loading datasets...")
    
    # Assume train/val split or single directory
    data_path = Path(args.data_path)
    
    if (data_path / 'train').exists() and (data_path / 'val').exists():
        # Separate train/val directories
        train_dataset = BrainTumorDataset(data_path / 'train', transform=train_transform)
        val_dataset = BrainTumorDataset(data_path / 'val', transform=val_transform, 
                                       class_to_idx=train_dataset.class_to_idx)
    else:
        # Single directory - need to split
        full_dataset = BrainTumorDataset(data_path, transform=train_transform)
        
        # 80-20 split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update validation dataset transform
        val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    # Get class information
    if hasattr(train_dataset, 'classes'):
        class_names = train_dataset.classes
        num_classes = len(class_names)
    else:
        class_names = train_dataset.dataset.classes
        num_classes = len(class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print(f"\nCreating {args.architecture} model...")
    model = BrainTumorClassifier(num_classes, args.architecture, args.pretrained)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.5, verbose=True)
    
    # Create trainer
    trainer = Trainer(model, device, num_classes, class_names)
    
    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        save_dir=output_dir
    )
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training history
    plot_training_history(history, output_dir / 'training_history.png')
    
    # Final validation for confusion matrix
    print("\nGenerating final evaluation...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc='Final evaluation'):
            data = data.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Create confusion matrix
    create_confusion_matrix(all_targets, all_preds, class_names, 
                          output_dir / 'confusion_matrix.png')
    
    print(f"\nTraining completed! Results saved to: {output_dir}")
    
    # Print final summary
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Model Architecture: {args.architecture}")
    print(f"Total Epochs: {args.epochs}")
    print(f"Dataset: {args.data_path}")
    print("="*60)


if __name__ == '__main__':
    main()