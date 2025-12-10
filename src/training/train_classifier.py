#!/usr/bin/env python3
"""
Brain Tumor Classification Training Module
Complete implementation with data loading, training, validation, and result saving.
Fixed version that handles missing classes properly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import os
import json
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrainTumorDataset(Dataset):
    """Custom dataset for brain tumor classification"""
    
    def __init__(self, data_path, transform=None, split='train'):
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        
        # Define class mapping - Note: using 'no_tumor' instead of 'notumor' to match folder structure
        self.classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Print dataset statistics
        self._print_dataset_stats()
        
    def _load_samples(self):
        samples = []
        split_path = self.data_path / self.split
        
        logger.info(f"Loading {self.split} dataset from: {split_path}")
        
        if not split_path.exists():
            logger.warning(f"Split directory not found: {split_path}")
            # Try to find images in the main directory structure
            return self._load_samples_alternative()
        
        for class_name in self.classes:
            class_path = split_path / class_name
            if class_path.exists():
                class_samples = []
                for img_name in class_path.iterdir():
                    if img_name.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                        label = self.class_to_idx[class_name]
                        class_samples.append((str(img_name), label))
                
                samples.extend(class_samples)
                logger.info(f"  {class_name}: {len(class_samples)} images")
        
        if not samples:
            logger.warning(f"No samples found in {split_path}")
            return self._load_samples_alternative()
        
        return samples
    
    def _load_samples_alternative(self):
        """Alternative method to load samples if standard structure doesn't exist"""
        samples = []
        
        # Look for any subdirectories that might contain images
        for potential_class_dir in self.data_path.iterdir():
            if potential_class_dir.is_dir():
                class_name = potential_class_dir.name.lower()
                
                # Check if this matches one of our expected classes
                matched_class = None
                for expected_class in self.classes:
                    if expected_class.lower() in class_name or class_name in expected_class.lower():
                        matched_class = expected_class
                        break
                
                if matched_class:
                    class_samples = []
                    for img_file in potential_class_dir.iterdir():
                        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                            label = self.class_to_idx[matched_class]
                            class_samples.append((str(img_file), label))
                    
                    samples.extend(class_samples)
                    logger.info(f"  Found {matched_class}: {len(class_samples)} images in {potential_class_dir}")
        
        return samples
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        if not self.samples:
            logger.warning(f"No samples found for {self.split} split!")
            return
        
        # Count samples per class
        class_counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        
        logger.info(f"\n{self.split.upper()} Dataset Statistics:")
        logger.info(f"Total samples: {len(self.samples)}")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.samples)} samples")
        
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            blank_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, label

class BrainTumorClassifier(nn.Module):
    """CNN model for brain tumor classification"""
    
    def __init__(self, num_classes=4, architecture='resnet18', pretrained=True):
        super(BrainTumorClassifier, self).__init__()
        
        self.architecture = architecture
        
        if architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        
        elif architecture == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            self.backbone.classifier[6] = nn.Linear(4096, num_classes)
        
        else:
            # Default to ResNet18
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def get_data_transforms(input_size=224, augment=True):
    """Define data transformations for training and validation"""
    
    if augment:
        train_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def create_data_loaders(data_path, batch_size=32, num_workers=0, input_size=224, augment=True):
    """Create train and validation data loaders"""
    
    train_transforms, val_transforms = get_data_transforms(input_size, augment)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = BrainTumorDataset(data_path, transform=train_transforms, split='train')
    
    # Try to create validation dataset, fall back to using training data if validation doesn't exist
    try:
        val_dataset = BrainTumorDataset(data_path, transform=val_transforms, split='val')
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset is empty")
    except:
        # Try 'test' split instead of 'val'
        try:
            val_dataset = BrainTumorDataset(data_path, transform=val_transforms, split='test')
            if len(val_dataset) == 0:
                raise ValueError("Test dataset is empty")
            logger.info("Using test split as validation data.")
        except:
            logger.warning("Validation/test dataset not found or empty. Using 20% of training data for validation.")
            # Split training dataset
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            
            # Apply validation transforms to validation split
            val_dataset.dataset.transform = val_transforms
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Validation loader: {len(val_loader)} batches")
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_samples += labels.size(0)
        total_correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        accuracy = 100.0 * total_correct / total_samples
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    return total_loss / len(train_loader), 100.0 * total_correct / total_samples

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validation")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), 100.0 * total_correct / total_samples, all_predictions, all_labels

def safe_classification_report(y_true, y_pred, class_names):
    """Create classification report handling missing classes"""
    try:
        present_classes = sorted(set(y_true) | set(y_pred))
        
        logger.info(f"Expected classes: {class_names}")
        logger.info(f"Classes present in data: {[class_names[i] for i in present_classes]}")
        
        if len(present_classes) < len(class_names):
            # Only use classes that are actually present
            present_class_names = [class_names[i] for i in present_classes]
            missing_classes = [class_names[i] for i in range(len(class_names)) if i not in present_classes]
            logger.warning(f"Missing classes in validation data: {missing_classes}")
            
            return classification_report(
                y_true, y_pred, 
                labels=present_classes,
                target_names=present_class_names, 
                digits=4, 
                zero_division=0
            )
        else:
            return classification_report(
                y_true, y_pred, 
                target_names=class_names, 
                digits=4, 
                zero_division=0
            )
    except Exception as e:
        logger.error(f"Error creating classification report: {e}")
        return f"Classification report unavailable due to error: {e}"

def save_training_plots(train_losses, val_losses, train_accuracies, val_accuracies, output_dir):
    """Save training history plots"""
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss smoothing
    if len(train_losses) > 5:
        try:
            from scipy.ndimage import uniform_filter1d
            smooth_train = uniform_filter1d(train_losses, size=5)
            smooth_val = uniform_filter1d(val_losses, size=5)
            ax3.plot(epochs, smooth_train, 'b-', label='Smoothed Training Loss', linewidth=2)
            ax3.plot(epochs, smooth_val, 'r-', label='Smoothed Validation Loss', linewidth=2)
        except ImportError:
            ax3.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            ax3.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    else:
        ax3.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax3.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax3.set_title('Smoothed Loss Curves', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning rate progression (if available)
    ax4.plot(epochs, train_accuracies, 'g-', linewidth=3, alpha=0.7)
    ax4.fill_between(epochs, train_accuracies, alpha=0.3, color='green')
    ax4.set_title('Training Accuracy Progression', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_confusion_matrix(predictions, labels, class_names, output_dir):
    """Save confusion matrix plot"""
    try:
        # Only use classes that are present in the data
        present_classes = sorted(set(labels) | set(predictions))
        present_class_names = [class_names[i] for i in present_classes]
        
        cm = confusion_matrix(labels, predictions, labels=present_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=present_class_names, yticklabels=present_class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error creating confusion matrix: {e}")

def save_results_summary(results, output_dir):
    """Save detailed results summary"""
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save as readable text file
    with open(output_dir / 'training_summary.txt', 'w') as f:
        f.write("BRAIN TUMOR CLASSIFICATION - TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Final Training Accuracy: {results['final_train_accuracy']:.2f}%\n")
        f.write(f"Final Validation Accuracy: {results['final_val_accuracy']:.2f}%\n")
        f.write(f"Best Validation Accuracy: {results['best_val_accuracy']:.2f}%\n")
        f.write(f"Best Epoch: {results['best_epoch']}\n")
        f.write(f"Total Training Time: {results['total_training_time']:.2f} seconds\n\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-" * 30 + "\n")
        f.write(results['classification_report'])

def train_model(data_path, output_dir, epochs=100, batch_size=32, learning_rate=0.001, 
                device='cpu', architecture='resnet18', input_size=224, augment=True, num_workers=0):
    """Main training function"""
    
    start_time = time.time()
    
    # Convert string device to torch device
    if isinstance(device, str):
        device = torch.device(device)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting brain tumor classification training...")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    logger.info(f"Architecture: {architecture}, Input size: {input_size}")
    
    try:
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            data_path, batch_size, num_workers, input_size, augment
        )
        
        if len(train_loader) == 0:
            raise ValueError("No training data found!")
        
        # Initialize model
        logger.info(f"Initializing {architecture} model...")
        model = BrainTumorClassifier(num_classes=4, architecture=architecture, pretrained=True)
        model.to(device)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
        
        # Training history
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_acc = 0.0
        best_epoch = 0
        best_model_path = output_dir / 'best_model.pth'
        
        logger.info(f"\nStarting training for {epochs} epochs...")
        print("=" * 80)
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation phase
            val_loss, val_acc, val_predictions, val_labels = validate_epoch(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step()
            
            # Store history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.2f}s):")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                
                # Save model checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accuracies': train_accuracies,
                    'val_accuracies': val_accuracies,
                    'class_names': ['glioma', 'meningioma', 'no_tumor', 'pituitary'],
                    'architecture': architecture
                }, best_model_path)
                
                print(f"  *** New best model saved! Validation accuracy: {best_val_acc:.2f}% ***")
            
            print("-" * 80)
        
        total_time = time.time() - start_time
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"Total training time: {total_time:.2f} seconds")
        
        # Generate final evaluation
        logger.info("Generating final evaluation...")
        
        # Load best model for final evaluation
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final validation
        final_val_loss, final_val_acc, final_predictions, final_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Calculate detailed metrics
        class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        
        # Get present classes for precision/recall calculation
        present_classes = sorted(set(final_labels) | set(final_predictions))
        
        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                final_labels, final_predictions, 
                average=None, 
                labels=present_classes,
                zero_division=0
            )
        except Exception as e:
            logger.warning(f"Could not calculate detailed metrics: {e}")
            precision = recall = f1 = support = [0] * len(present_classes)
        
        # Use the safe classification report
        classification_rep = safe_classification_report(
            final_labels, final_predictions, class_names
        )
        
        # Save results
        results = {
            'model_architecture': architecture,
            'total_epochs': epochs,
            'best_epoch': best_epoch,
            'final_train_accuracy': train_accuracies[-1] if train_accuracies else 0,
            'final_val_accuracy': final_val_acc,
            'best_val_accuracy': best_val_acc,
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_val_loss': final_val_loss,
            'total_training_time': total_time,
            'class_names': class_names,
            'present_classes': [class_names[i] for i in present_classes],
            'precision': precision.tolist() if hasattr(precision, 'tolist') else precision,
            'recall': recall.tolist() if hasattr(recall, 'tolist') else recall,
            'f1_score': f1.tolist() if hasattr(f1, 'tolist') else f1,
            'support': support.tolist() if hasattr(support, 'tolist') else support,
            'classification_report': classification_rep,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }
        }
        
        # Save all results
        save_results_summary(results, output_dir)
        save_training_plots(train_losses, val_losses, train_accuracies, val_accuracies, output_dir)
        save_confusion_matrix(final_predictions, final_labels, class_names, output_dir)
        
        logger.info(f"All results saved to: {output_dir}")
        logger.info(f"Best model saved as: {best_model_path}")
        
        print(f"\nFinal Results:")
        print(f"- Model: {best_model_path}")
        print(f"- Training plots: {output_dir}/training_history.png")
        print(f"- Confusion matrix: {output_dir}/confusion_matrix.png")
        print(f"- Detailed results: {output_dir}/training_results.json")
        print(f"- Summary: {output_dir}/training_summary.txt")
        
        return model, results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Test the training function
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Path to dataset')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--architecture', default='resnet18', help='Model architecture')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        architecture=args.architecture
    )