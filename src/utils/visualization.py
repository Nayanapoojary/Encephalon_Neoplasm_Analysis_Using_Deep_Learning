import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import cv2
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_training_history(history, title="Training History", save_path=None):
    """
    Plot training history with loss and accuracy curves
    
    Args:
        history (dict): Training history dictionary
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.7)
    axes[0].plot(history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy (or other metric)
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy', color='blue', alpha=0.7)
        axes[1].plot(history['val_acc'], label='Validation Accuracy', color='red', alpha=0.7)
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
    elif 'train_dice' in history:
        axes[1].plot(history['train_dice'], label='Train Dice', color='blue', alpha=0.7)
        axes[1].plot(history['val_dice'], label='Validation Dice', color='red', alpha=0.7)
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Dice Score')
    
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix with nice formatting
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add accuracy for each class
    for i, class_name in enumerate(class_names):
        accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        plt.text(i + 0.5, i - 0.3, f'{accuracy:.2%}', 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_brain_scan(image, mask=None, prediction=None, title="Brain Scan", save_path=None):
    """
    Visualize brain scan with optional mask and prediction overlay
    
    Args:
        image: Input brain scan image
        mask: Ground truth mask (optional)
        prediction: Predicted mask (optional)
        title: Plot title
        save_path: Path to save the plot
    """
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().squeeze()
    
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy().squeeze()
    
    # Determine number of subplots
    num_plots = 1
    if mask is not None:
        num_plots += 1
    if prediction is not None:
        num_plots += 1
    if mask is not None and prediction is not None:
        num_plots += 1  # For overlay comparison
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Original image
    if len(image.shape) == 3:
        axes[plot_idx].imshow(image)
    else:
        axes[plot_idx].imshow(image, cmap='gray')
    axes[plot_idx].set_title('Original Image')
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # Ground truth mask
    if mask is not None:
        axes[plot_idx].imshow(mask, cmap='Reds', alpha=0.8)
        axes[plot_idx].set_title('Ground Truth')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Prediction mask
    if prediction is not None:
        axes[plot_idx].imshow(prediction, cmap='Blues', alpha=0.8)
        axes[plot_idx].set_title('Prediction')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Overlay comparison
    if mask is not None and prediction is not None:
        # Create RGB overlay
        overlay = np.zeros((*mask.shape, 3))
        overlay[:, :, 0] = mask  # Red for ground truth
        overlay[:, :, 2] = prediction  # Blue for prediction
        
        axes[plot_idx].imshow(overlay, alpha=0.7)
        axes[plot_idx].set_title('Overlay (Red: GT, Blue: Pred)')
        axes[plot_idx].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_segmentation_results(images, masks, predictions, class_names=None, num_samples=8):
    """
    Visualize segmentation results in a grid
    
    Args:
        images: Input images
        masks: Ground truth masks
        predictions: Predicted masks
        class_names: Names of classes
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(4, num_samples, figsize=(2*num_samples, 8))
    
    for i in range(num_samples):
        # Convert tensors if needed
        img = images[i]
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
        
        mask = masks[i]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy().squeeze()
        
        pred = predictions[i]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy().squeeze()
        
        # Denormalize image if needed
        if img.max() <= 1.0 and len(img.shape) == 3:
            img = np.clip(img, 0, 1)
        
        # Original image
        axes[0, i].imshow(img if len(img.shape) == 3 else img, cmap='gray' if len(img.shape) == 2 else None)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Ground truth
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Ground Truth {i+1}')
        axes[1, i].axis('off')
        
        # Prediction
        axes[2, i].imshow(pred, cmap='gray')
        axes[2, i].set_title(f'Prediction {i+1}')
        axes[2, i].axis('off')
        
        # Overlay
        if len(img.shape) == 3:
            overlay = img.copy()
        else:
            overlay = np.stack([img, img, img], axis=2)
        
        # Add red outline for prediction
        pred_binary = (pred > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        overlay_with_contour = overlay.copy()
        cv2.drawContours(overlay_with_contour, contours, -1, (1, 0, 0), 2)
        
        axes[3, i].imshow(overlay_with_contour)
        axes[3, i].set_title(f'Overlay {i+1}')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results_dict, metric='accuracy', title="Model Comparison"):
    """
    Create bar plot comparing different models
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
        metric: Metric to compare
        title: Plot title
    """
    models = list(results_dict.keys())
    values = [results_dict[model].get(metric, 0) for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(title)
    plt.ylabel(metric.capitalize())
    plt.xlabel('Model')
    plt.ylim(0, max(values) * 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_interactive_results_dashboard(classification_results, segmentation_results, view_results):
    """
    Create interactive dashboard using Plotly
    
    Args:
        classification_results: Classification model results
        segmentation_results: Segmentation model results  
        view_results: View classification results
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Classification Accuracy', 'Segmentation Dice Score', 
                       'View Classification Accuracy', 'Training Progress'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Classification results
    if classification_results:
        models = list(classification_results.keys())
        accuracies = [classification_results[model].get('accuracy', 0) for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='Classification', marker_color='blue'),
            row=1, col=1
        )
    
    # Segmentation results
    if segmentation_results:
        models = list(segmentation_results.keys())
        dice_scores = [segmentation_results[model].get('dice', 0) for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=dice_scores, name='Segmentation', marker_color='red'),
            row=1, col=2
        )
    
    # View classification results
    if view_results:
        models = list(view_results.keys())
        accuracies = [view_results[model].get('accuracy', 0) for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='View Classification', marker_color='green'),
            row=2, col=1
        )
    
    # Training progress (example with dummy data)
    epochs = list(range(1, 51))
    train_acc = [0.5 + 0.4 * (1 - np.exp(-epoch/10)) + 0.05 * np.random.random() for epoch in epochs]
    val_acc = [0.4 + 0.35 * (1 - np.exp(-epoch/12)) + 0.08 * np.random.random() for epoch in epochs]
    
    fig.add_trace(
        go.Scatter(x=epochs, y=train_acc, name='Train Accuracy', line=dict(color='blue')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_acc, name='Validation Accuracy', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Brain Tumor Detection Results Dashboard",
        title_x=0.5,
        height=800,
        showlegend=True
    )
    
    return fig

def save_predictions_plot(images, true_labels, pred_labels, class_names, save_path, num_samples=16):
    """
    Save a plot showing prediction examples
    
    Args:
        images: Input images
        true_labels: True labels
        pred_labels: Predicted labels  
        class_names: List of class names
        save_path: Path to save the plot
        num_samples: Number of samples to show
    """
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        img = images[i]
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
        
        # Denormalize if needed
        if img.max() <= 1.0:
            img = np.clip(img, 0, 1)
        
        axes[i].imshow(img if len(img.shape) == 3 else img, cmap='gray' if len(img.shape) == 2 else None)
        
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        title = f'True: {true_class}\nPred: {pred_class}'
        
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_severity_distribution(severity_predictions, title="Tumor Severity Distribution", save_path=None):
    """
    Plot distribution of tumor severity predictions
    
    Args:
        severity_predictions: List of (severity, percentage) tuples
        title: Plot title
        save_path: Path to save the plot
    """
    severities = [pred[0] for pred in severity_predictions]
    percentages = [pred[1] * 100 for pred in severity_predictions]  # Convert to percentage
    
    # Count severity levels
    severity_counts = {}
    severity_percentages = {}
    
    for severity, percentage in zip(severities, percentages):
        if severity not in severity_counts:
            severity_counts[severity] = 0
            severity_percentages[severity] = []
        severity_counts[severity] += 1
        severity_percentages[severity].append(percentage)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Severity distribution pie chart
    labels = list(severity_counts.keys())
    sizes = list(severity_counts.values())
    colors = ['lightgreen', 'orange', 'red'][:len(labels)]
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Severity Level Distribution')
    
    # Tumor area percentage box plot
    severity_names = list(severity_percentages.keys())
    severity_data = [severity_percentages[name] for name in severity_names]
    
    box_plot = ax2.boxplot(severity_data, labels=severity_names, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors[:len(severity_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('Tumor Area Percentage by Severity')
    ax2.set_ylabel('Tumor Area (%)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_view_classification(images, predictions, true_labels, view_names, num_samples=12):
    """
    Visualize view classification results
    
    Args:
        images: Input brain scan images
        predictions: Predicted view labels
        true_labels: True view labels
        view_names: List of view names ['axial', 'sagittal', 'coronal']
        num_samples: Number of samples to show
    """
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        img = images[i]
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
        
        # Denormalize if needed
        if img.max() <= 1.0 and len(img.shape) == 3:
            img = np.clip(img, 0, 1)
        
        axes[i].imshow(img if len(img.shape) == 3 else img, cmap='gray' if len(img.shape) == 2 else None)
        
        true_view = view_names[true_labels[i]]
        pred_view = view_names[predictions[i]]
        
        color = 'green' if true_labels[i] == predictions[i] else 'red'
        title = f'True: {true_view}\nPred: {pred_view}'
        
        axes[i].set_title(title, color=color, fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Brain Scan View Classification Results', fontsize=16)
    plt.tight_layout()
    plt.show()

def create_comprehensive_report(classification_results, segmentation_results, view_results, save_path=None):
    """
    Create a comprehensive HTML report of all results
    
    Args:
        classification_results: Dictionary of classification results
        segmentation_results: Dictionary of segmentation results
        view_results: Dictionary of view classification results
        save_path: Path to save HTML report
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brain Tumor Detection - Comprehensive Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { text-align: center; color: #333; }
            .section { margin: 30px 0; }
            .metrics-table { border-collapse: collapse; width: 100%; }
            .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            .metrics-table th { background-color: #f2f2f2; }
            .highlight { background-color: #e7f3ff; }
            .best-score { font-weight: bold; color: #2e8b57; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Brain Tumor Detection System</h1>
            <h2>Comprehensive Performance Report</h2>
        </div>
    """
    
    # Classification Results
    if classification_results:
        html_content += """
        <div class="section">
            <h3>Tumor Classification Results</h3>
            <table class="metrics-table">
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Parameters</th>
                </tr>
        """
        
        best_acc = max(results.get('accuracy', 0) for results in classification_results.values())
        
        for model, results in classification_results.items():
            acc = results.get('accuracy', 0)
            precision = results.get('precision', 0)
            recall = results.get('recall', 0)
            f1 = results.get('f1', 0)
            params = results.get('parameters', 0)
            
            acc_class = 'best-score' if acc == best_acc else ''
            
            html_content += f"""
                <tr>
                    <td>{model}</td>
                    <td class="{acc_class}">{acc:.4f}</td>
                    <td>{precision:.4f}</td>
                    <td>{recall:.4f}</td>
                    <td>{f1:.4f}</td>
                    <td>{params:,}</td>
                </tr>
            """
        
        html_content += "</table></div>"
    
    # Segmentation Results
    if segmentation_results:
        html_content += """
        <div class="section">
            <h3>Tumor Segmentation Results</h3>
            <table class="metrics-table">
                <tr>
                    <th>Model</th>
                    <th>Dice Score</th>
                    <th>IoU Score</th>
                    <th>Sensitivity</th>
                    <th>Specificity</th>
                    <th>Parameters</th>
                </tr>
        """
        
        best_dice = max(results.get('dice', 0) for results in segmentation_results.values())
        
        for model, results in segmentation_results.items():
            dice = results.get('dice', 0)
            iou = results.get('iou', 0)
            sensitivity = results.get('sensitivity', 0)
            specificity = results.get('specificity', 0)
            params = results.get('parameters', 0)
            
            dice_class = 'best-score' if dice == best_dice else ''
            
            html_content += f"""
                <tr>
                    <td>{model}</td>
                    <td class="{dice_class}">{dice:.4f}</td>
                    <td>{iou:.4f}</td>
                    <td>{sensitivity:.4f}</td>
                    <td>{specificity:.4f}</td>
                    <td>{params:,}</td>
                </tr>
            """
        
        html_content += "</table></div>"
    
    # View Classification Results
    if view_results:
        html_content += """
        <div class="section">
            <h3>View Classification Results</h3>
            <table class="metrics-table">
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Axial Precision</th>
                    <th>Sagittal Precision</th>
                    <th>Coronal Precision</th>
                    <th>Parameters</th>
                </tr>
        """
        
        for model, results in view_results.items():
            acc = results.get('accuracy', 0)
            axial_prec = results.get('axial_precision', 0)
            sagittal_prec = results.get('sagittal_precision', 0)
            coronal_prec = results.get('coronal_precision', 0)
            params = results.get('parameters', 0)
            
            html_content += f"""
                <tr>
                    <td>{model}</td>
                    <td>{acc:.4f}</td>
                    <td>{axial_prec:.4f}</td>
                    <td>{sagittal_prec:.4f}</td>
                    <td>{coronal_prec:.4f}</td>
                    <td>{params:,}</td>
                </tr>
            """
        
        html_content += "</table></div>"
    
    # Summary and Recommendations
    html_content += """
    <div class="section">
        <h3>Summary and Recommendations</h3>
        <div class="highlight" style="padding: 20px; border-radius: 5px;">
            <h4>Best Performing Models:</h4>
            <ul>
    """
    
    if classification_results:
        best_clf_model = max(classification_results.items(), key=lambda x: x[1].get('accuracy', 0))
        html_content += f"<li><strong>Classification:</strong> {best_clf_model[0]} (Accuracy: {best_clf_model[1].get('accuracy', 0):.4f})</li>"
    
    if segmentation_results:
        best_seg_model = max(segmentation_results.items(), key=lambda x: x[1].get('dice', 0))
        html_content += f"<li><strong>Segmentation:</strong> {best_seg_model[0]} (Dice: {best_seg_model[1].get('dice', 0):.4f})</li>"
    
    if view_results:
        best_view_model = max(view_results.items(), key=lambda x: x[1].get('accuracy', 0))
        html_content += f"<li><strong>View Classification:</strong> {best_view_model[0]} (Accuracy: {best_view_model[1].get('accuracy', 0):.4f})</li>"
    
    html_content += """
            </ul>
            <h4>Hardware Recommendations for Intel i5:</h4>
            <ul>
                <li>Use EfficientNet or MobileNet variants for faster inference</li>
                <li>Consider model quantization for production deployment</li>
                <li>Batch size of 8-16 recommended for optimal memory usage</li>
                <li>Enable mixed precision training if GPU available</li>
            </ul>
        </div>
    </div>
    </body>
    </html>
    """
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(html_content)
        print(f"Comprehensive report saved to {save_path}")
    
    return html_content

def visualize_model_architecture(model, input_size=(1, 3, 224, 224), save_path=None):
    """
    Visualize model architecture (requires torchviz)
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        save_path: Path to save architecture diagram
    """
    try:
        from torchviz import make_dot # type: ignore
        
        # Create dummy input
        dummy_input = torch.randn(input_size)
        
        # Forward pass
        if hasattr(model, '__call__'):
            output = model(dummy_input)
        
        # Create visualization
        dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        
        if save_path:
            dot.render(save_path, format='png', cleanup=True)
            print(f"Model architecture saved to {save_path}.png")
        
        return dot
    
    except ImportError:
        print("torchviz not available. Install with: pip install torchviz")
        return None

def create_prediction_examples_grid(images, predictions, confidences, class_names, 
                                  num_examples=16, save_path=None):
    """
    Create a grid showing prediction examples with confidence scores
    
    Args:
        images: Input images
        predictions: Model predictions
        confidences: Prediction confidence scores
        class_names: List of class names
        num_examples: Number of examples to show
        save_path: Path to save the grid
    """
    num_examples = min(num_examples, len(images))
    cols = 4
    rows = (num_examples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if num_examples > 1 else [axes]
    
    for i in range(num_examples):
        img = images[i]
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
        
        # Denormalize if needed
        if img.max() <= 1.0 and len(img.shape) == 3:
            img = np.clip(img, 0, 1)
        
        axes[i].imshow(img if len(img.shape) == 3 else img, cmap='gray' if len(img.shape) == 2 else None)
        
        pred_class = class_names[predictions[i]]
        confidence = confidences[i]
        
        # Color based on confidence
        if confidence > 0.9:
            color = 'green'
        elif confidence > 0.7:
            color = 'orange'
        else:
            color = 'red'
        
        title = f'{pred_class}\n{confidence:.2%} confidence'
        axes[i].set_title(title, color=color, fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_examples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Prediction Examples with Confidence Scores', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Test training history plot
    dummy_history = {
        'train_loss': [1.0, 0.8, 0.6, 0.4, 0.3],
        'val_loss': [1.2, 0.9, 0.7, 0.5, 0.4],
        'train_acc': [0.3, 0.5, 0.7, 0.8, 0.85],
        'val_acc': [0.25, 0.45, 0.65, 0.75, 0.8]
    }
    
    plot_training_history(dummy_history, title="Test Training History")
    
    print("Visualization utilities test completed!")