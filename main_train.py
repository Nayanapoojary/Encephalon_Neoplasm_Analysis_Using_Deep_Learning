#!/usr/bin/env python3
"""
Main training script for Encephalon Neoplasm Detection Project.
Supports classification and segmentation tasks.
Includes training, validation, and testing.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train models for brain tumor detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Task selection
    parser.add_argument('--task', type=str, required=True,
                        choices=['classification', 'segmentation'],
                        help='Task type: classification or segmentation')

    # Data arguments
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save outputs (models, plots, etc.)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')

    # Model arguments
    parser.add_argument('--architecture', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'efficientnet_b0', 'vgg16'],
                        help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size')

    # Data augmentation
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='Disable data augmentation')

    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Advanced options
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only run validation (requires --resume)')
    parser.add_argument('--test', action='store_true',
                        help='Run testing after training')

    return parser.parse_args()


def setup_environment(args):
    """Setup environment and check prerequisites."""
    import torch
    import numpy as np
    import random

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Environment Setup:")
    print(f"  Device: {device}")
    print(f"  Random seed: {args.seed}")
    print(f"  Data path: {data_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Task: {args.task}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return device, output_dir, data_path


def run_testing(model, device, data_path, args):
    """Run testing on unseen test dataset."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report, confusion_matrix
    import torch
    import os

    test_dir = os.path.join(str(data_path), "test")
    if not os.path.exists(test_dir):
        print(f"Test directory not found at: {test_dir}")
        return

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nTest Set Evaluation:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def main():
    """Main training function."""
    args = parse_arguments()
    device, output_dir, data_path = setup_environment(args)

    # Import task-specific training modules
    if args.task == 'classification':
        print("\n" + "="*60)
        print("BRAIN TUMOR CLASSIFICATION TRAINING")
        print("="*60)

        try:
            from src.training.train_classifier import train_model

            model = train_model(
                data_path=str(data_path),
                output_dir=str(output_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=device
            )

        except ImportError as e:
            print(f"Error importing classification module: {e}")
            print("Make sure src/training/train_classifier.py exists")
            return 1

    elif args.task == 'segmentation':
        print("\n" + "="*60)
        print("BRAIN TUMOR SEGMENTATION TRAINING")
        print("="*60)

        try:
            from src.training.train_segmentation import train_model as train_segmentation_model

            model = train_segmentation_model(
                data_path=str(data_path),
                output_dir=str(output_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=device
            )

        except ImportError as e:
            print(f"Error importing segmentation module: {e}")
            print("Creating placeholder segmentation trainer...")
            model = None

    else:
        raise ValueError(f"Unknown task: {args.task}")

    # === Optional Testing Phase ===
    if args.test and model is not None:
        run_testing(model, device, data_path, args)

    print(f"\nTraining completed! Check results in: {output_dir}")
    return 0


def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'torchvision', 'PIL', 'numpy',
        'matplotlib', 'seaborn', 'sklearn', 'tqdm'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


if __name__ == '__main__':
    print("Encephalon Neoplasm Detection - Main Training Script")
    print("="*60)

    if not check_requirements():
        print("Please install missing requirements before continuing.")
        sys.exit(1)

    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
