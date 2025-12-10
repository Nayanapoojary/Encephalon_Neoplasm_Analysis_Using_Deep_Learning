# BRIAC2025 Competition

A comprehensive machine learning framework for classification and segmentation tasks in the BRIAC2025 competition.

## 📁 Project Structure

```
├── data/
│   └── briac2025/
│       ├── classification_task/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── segmentation_task/
│           ├── train/
│           ├── val/
│           └── test/
├── src/
│   ├── models/
│   │   ├── classification_model.py
│   │   ├── segmentation_model.py
│   │   └── view_classifier.py
│   ├── training/
│   │   ├── train_classifier.py
│   │   ├── train_segmentation.py
│   │   └── train_view_classifier.py
│   ├── inference/
│   │   ├── app.py
│   │   └── predict.py
│   └── utils/
│       ├── config.py
│       ├── data_utils.py
│       ├── metrics.py
│       └── visualization.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── results/
├── main_train.py
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd briac2025
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Data Preparation

Organize your data in the following structure:

**For Classification:**
```
data/briac2025/classification_task/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
├── val/
│   └── ...
└── test/
    └── ...
```

**For Segmentation:**
```
data/briac2025/segmentation_task/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── masks/
│       ├── image1.png
│       └── image2.png
├── val/
│   └── ...
└── test/
    └── ...
```

## 🏋️ Training

### Classification Task

```bash
py main_train.py --task classification --data-path "C:\Users\navee\Downloads\encephalon_neoplasm_major_project\data\brisc2025\classification_task" --output-dir "C:\Users\navee\Downloads\encephalon_neoplasm_major_project\results" --epochs 100 --batch-size 32 --learning-rate 0.001

```
py analyze_results.py --results-dir "C:\Users\navee\Downloads\encephalon_neoplasm_major_project\results"
### Segmentation Task

```bash
python main_train.py \
    --task segmentation \
    --data-path data/briac2025/segmentation_task \
    --output-dir results/segmentation \
    --epochs 150 \
    --batch-size 16 \
    --learning-rate 0.0001
```

### Using Configuration Files

Create a YAML configuration file:

```yaml
# config/classification.yaml
task: classification
model:
  name: resnet50
  num_classes: 10
  pretrained: true
  dropout: 0.5

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  scheduler:
    name: cosine
    T_max: 100

data:
  image_size: 224
  normalize: true
  augmentation:
    horizontal_flip: 0.5
    rotation: 15
    color_jitter: 0.2
```

Then run:
```bash
python main_train.py \
    --config config/classification.yaml \
    --data-path data/briac2025/classification_task
```

## 🔮 Inference

### Batch Prediction

```bash
python -m src.inference.predict \
    --model-path results/classification/best_model.pth \
    --data-path data/test_images \
    --output-path results/predictions.csv \
    --task classification
```

### Interactive Web App

```bash
python -m src.inference.app
```

Then open your browser to `http://localhost:8501` to use the interactive interface.

## 📊 Evaluation and Monitoring

### View Training Progress

```bash
tensorboard --logdir results/
```

### Evaluate Model Performance

```bash
python -m src.utils.metrics \
    --model-path results/classification/best_model.pth \
    --data-path data/briac2025/classification_task \
    --task classification
```

## 📓 Jupyter Notebooks

Explore the provided notebooks for detailed analysis:

- `notebooks/data_exploration.ipynb`: Dataset analysis and visualization
- `notebooks/model_training.ipynb`: Interactive model training and experimentation
- `notebooks/results_analysis.ipynb`: Results visualization and comparison

## ⚙️ Configuration Options

### Model Configuration

**Classification Models:**
- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0-B7)
- Vision Transformer (ViT)
- ConvNeXt

**Segmentation Models:**
- U-Net
- DeepLabV3+
- PSPNet
- FPN

### Training Configuration

**Optimizers:**
- Adam
- AdamW
- SGD
- RMSprop

**Schedulers:**
- Cosine Annealing
- Step LR
- Exponential LR
- Polynomial LR

**Loss Functions:**
- CrossEntropy (Classification)
- Focal Loss (Classification)
- Dice Loss (Segmentation)
- Combined Loss (Segmentation)

## 🔧 Advanced Usage

### Custom Model Implementation

```python
from src.models.base_model import BaseModel
import torch.nn as nn

class CustomClassifier(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = self._create_backbone()
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
```

### Custom Loss Function

```python
from src.training.losses import BaseLoss
import torch.nn.functional as F

class CustomLoss(BaseLoss):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions, targets):
        return F.cross_entropy(predictions, targets, weight=self.weight)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Training:**
   - Increase number of workers
   - Use faster data loading
   - Enable pin_memory

3. **Poor Performance:**
   - Check data augmentation
   - Adjust learning rate
   - Use learning rate scheduling

### Debug Mode

Enable debug logging:
```bash
python main_train.py --log-level DEBUG [other args]
```

## 📈 Performance Tips

1. **Data Loading Optimization:**
   ```python
   # Use multiple workers
   num_workers = min(8, os.cpu_count())
   
   # Enable memory pinning
   pin_memory = True
   
   # Use persistent workers (PyTorch 1.7+)
   persistent_workers = True
   ```

2. **Mixed Precision Training:**
   ```python
   # Enable in config
   mixed_precision: true
   ```

3. **Model Compilation (PyTorch 2.0+):**
   ```python
   model = torch.compile(model)
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BRIAC2025 Competition organizers
- PyTorch and torchvision communities
- Open source contributors

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Email: support@briac2025.com
- Documentation: [Link to docs]

---

**Happy Training! 🎯**