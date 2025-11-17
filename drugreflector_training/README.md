# DrugReflector Training Module

This module provides training utilities for DrugReflector base models following the methodology described in **Science 2025 Supplementary Materials**.

## Overview

The training module implements:

- **3-Fold Ensemble Training**: Train 3 independent models on different data splits
- **Focal Loss**: Address class imbalance (γ=2.0, from SI Page 3)
- **Cosine Annealing with Warm Restarts**: Learning rate scheduling (from SI Pages 3-4)
- **Signature Normalization**: Clip to [-2, 2] with std=1 (from SI Page 3)
- **Compatible Checkpoints**: Models can be loaded by the original `DrugReflector` class

## File Structure

```
drugreflector_training/
├── __init__.py           - Module initialization
├── dataset.py            - PyTorch Dataset for LINCS data
├── losses.py             - Focal Loss implementation
├── preprocessing.py      - Data normalization and fold splitting
├── trainer.py            - Main training logic
├── evaluator.py          - Model evaluation utilities
└── visualization.py      - Training curve plotting

scripts/
├── prepare_data.py       - Prepare training data from LINCS files
├── train.py              - Main training script
└── inference_example.py  - Example inference usage
```

## Installation

The training module requires the base DrugReflector package plus PyTorch:

```bash
# Install dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm

# The module expects to import from the parent drugreflector package
# Make sure the drugreflector package is in your Python path
```

## Quick Start

### 1. Prepare Training Data

If you have raw LINCS expression data:

```bash
python scripts/prepare_data.py \
    --expression-file /path/to/expression.npy \
    --metadata-file /path/to/metadata.csv \
    --output-file ./data/training_data.pkl
```

Expected data format:
- **Expression file**: Numpy array (.npy) or CSV with shape (n_samples, 978)
- **Metadata file**: CSV with at least a `pert_id` column for compound IDs

### 2. Train Models

Train the full 3-fold ensemble:

```bash
python scripts/train.py \
    --data-file ./data/training_data.pkl \
    --output-dir ./trained_models \
    --epochs 50
```

For quick testing (single fold, 10 epochs):

```bash
python scripts/train.py \
    --data-file ./data/training_data.pkl \
    --output-dir ./test_models \
    --epochs 10 \
    --single-fold 0
```

### 3. Use Trained Models

The trained models are compatible with the original `DrugReflector` class:

```python
from drugreflector import DrugReflector
import pandas as pd

# Load models
model = DrugReflector(checkpoint_paths=[
    './trained_models/model_fold_0.pt',
    './trained_models/model_fold_1.pt',
    './trained_models/model_fold_2.pt'
])

# Create v-scores (example)
vscores = pd.Series([1.2, -0.8, 0.5, ...], index=['GENE1', 'GENE2', ...])

# Make predictions
predictions = model.predict(vscores, n_top=100)
```

Or use the example script:

```bash
python scripts/inference_example.py \
    --model-dir ./trained_models \
    --top-k 100
```

## Training Parameters

### Default Configuration (from Paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| Initial LR | 0.0139 | SI Table S5 |
| Min LR | 0.00001 | SI Page 3 |
| Weight Decay | 1e-5 | SI Table S5 |
| Dropout | 0.64 | SI Table S5 |
| Focal γ | 2.0 | SI Page 3 |
| T_0 (warm restart) | 20 | SI Table S5 |
| Epochs | 50 | SI Page 3 |
| Batch Size | 256 | - |

### Customizing Training

```bash
python scripts/train.py \
    --data-file ./data/training_data.pkl \
    --output-dir ./custom_models \
    --epochs 100 \
    --batch-size 512 \
    --learning-rate 0.02 \
    --focal-gamma 2.5 \
    --device cuda
```

## Model Architecture

The trained models use the `nnFC` architecture from the base package:

```
Input (978 landmark genes)
    ↓
Linear(978 → 1024) + BatchNorm + ReLU + Dropout(0.64)
    ↓
Linear(1024 → 2048) + BatchNorm + ReLU + Dropout(0.64)
    ↓
Linear(2048 → 9597 compounds)
    ↓
Output (logits)
```

Total parameters: ~22.7M

## Expected Performance

Based on Science 2025 SI Page 24:

| Dataset | Top 1% Recall |
|---------|---------------|
| CMap Touchstone | 0.46-0.50 |
| sciPlex3 | 0.30-0.35 |
| Internal | 0.60-0.65 |

Training time (estimated):
- **Single model**: 6-8 hours (RTX 3090)
- **Full ensemble**: 18-24 hours (RTX 3090)

## Checkpoint Format

Trained checkpoints are compatible with the original `EnsembleModel` class. Each checkpoint contains:

```python
{
    'model_state_dict': {...},        # PyTorch state dict
    'fold_id': 0,                     # Fold identifier
    'history': {...},                 # Training history
    'dimensions': {
        'input_size': 978,
        'output_size': 9597,
        'input_names': [...],         # Gene names
        'output_names': [...]         # Compound names
    },
    'params_init': {                  # Model initialization params
        'model_init_params': {
            'torch_init_params': {...}
        }
    }
}
```

## API Usage

### Training Programmatically

```python
from drugreflector_training import DrugReflectorTrainer
import pickle
from pathlib import Path

# Load data
with open('training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)

# Initialize trainer
trainer = DrugReflectorTrainer(
    device='cuda',
    initial_lr=0.0139,
    focal_gamma=2.0,
    batch_size=256,
    num_epochs=50
)

# Train ensemble
output_dir = Path('./models')
models, histories = trainer.train_ensemble(training_data, output_dir)
```

### Evaluation

```python
from drugreflector_training import DrugReflectorEvaluator

# Create evaluator
evaluator = DrugReflectorEvaluator(models, device='cuda')

# Evaluate
results = evaluator.evaluate(X_test, y_test, compound_names)

print(f"Top 1% Recall: {results['top1_percent_recall']:.4f}")
print(f"Top-10 Accuracy: {results['top10_accuracy']:.4f}")
```

## Training Data Format

The training data pickle file should contain:

```python
{
    'X': np.ndarray,              # Expression matrix (n_samples, 978)
    'y': np.ndarray,              # Compound labels (n_samples,)
    'folds': np.ndarray,          # Fold assignments (n_samples,)
    'compound_names': list,       # List of compound identifiers
    'gene_names': list,           # List of gene names
    'metadata': pd.DataFrame      # Optional metadata
}
```

## Troubleshooting

### GPU Out of Memory

Reduce batch size:
```bash
python scripts/train.py --batch-size 128
```

Or use CPU (much slower):
```bash
python scripts/train.py --device cpu
```

### Import Errors

Ensure the parent `drugreflector` package is in your Python path:

```python
import sys
sys.path.append('/path/to/drugreflector')
```

### Model Loading Issues

The checkpoints must be in the format expected by `EnsembleModel`. If you encounter loading errors, verify:

1. Checkpoint has all required keys (see Checkpoint Format above)
2. Model architecture matches (hidden_dims=[1024, 2048], dropout_p=0.64)
3. Gene names and compound names are lists (not pandas Index)

## Citation

If you use this training code, please cite:

```bibtex
@article{demeo2025drugreflector,
  title={Active learning framework leveraging transcriptomics identifies modulators of disease phenotypes},
  author={DeMeo, Benjamin and others},
  journal={Science},
  year={2025},
  doi={10.1126/science.adi8577}
}
```

## License

This code follows the same license as the main DrugReflector package. See LICENSE.txt in the parent directory.

## Support

For issues related to:
- **Training code**: Check this README and example scripts
- **Original DrugReflector**: See main package README
- **Model loading**: Verify checkpoint format compatibility

## Changelog

### Version 1.0.0
- Initial release
- 3-fold ensemble training
- Focal Loss implementation
- Compatible checkpoint format
- Integrated with original DrugReflector inference