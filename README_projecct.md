# DrugReflector - Complete Package with Training

DrugReflector for compound ranking predictions from gene expression signatures, now with **training capabilities**.

## Project Structure

```
drugreflector/
├── drugreflector/                 # Original inference package
│   ├── drug_reflector.py         # Main DrugReflector class
│   ├── ensemble_model.py         # Ensemble model loader
│   ├── models.py                 # nnFC neural network architecture
│   ├── utils.py                  # DrugReflector-specific utilities
│   └── predict.py                # CLI for predictions
│
├── drugreflector_training/        # NEW: Training module
│   ├── dataset.py                # PyTorch Dataset
│   ├── losses.py                 # Focal Loss
│   ├── preprocessing.py          # Data normalization
│   ├── trainer.py                # Training logic
│   ├── evaluator.py              # Model evaluation
│   ├── visualization.py          # Training plots
│   └── README.md                 # Training documentation
│
├── signature_refinement/          # Signature refinement tools
│   └── signature_refinement.py
│
├── scripts/                       # NEW: Training scripts
│   ├── prepare_data.py           # Data preparation
│   ├── train.py                  # Main training script
│   └── inference_example.py      # Inference examples
│
├── utils.py                       # Shared utilities (v-scores, etc.)
├── checkpoints/                   # Pre-trained model checkpoints
├── README.md                      # This file
└── LICENSE.txt
```

## Two Use Cases

### 1. Inference with Pre-trained Models (Original)

Use the published pre-trained models for drug screening:

```python
from drugreflector import DrugReflector
import pandas as pd

# Load pre-trained models from Zenodo
model = DrugReflector(checkpoint_paths=[
    'checkpoints/model_fold_0.pt',  # Download from Zenodo
    'checkpoints/model_fold_1.pt',
    'checkpoints/model_fold_2.pt'
])

# Compute v-scores from your data
vscores = pd.Series([1.2, -0.8, ...], index=['TP53', 'EGFR', ...])

# Predict compounds
predictions = model.predict(vscores, n_top=100)
```

**See the main [README.md](README.md) for detailed inference usage.**

### 2. Training Your Own Models (New)

Train custom DrugReflector models on your own data:

```bash
# 1. Prepare training data
python scripts/prepare_data.py \
    --expression-file /path/to/lincs_expression.npy \
    --metadata-file /path/to/lincs_metadata.csv \
    --output-file ./data/training_data.pkl

# 2. Train models
python scripts/train.py \
    --data-file ./data/training_data.pkl \
    --output-dir ./my_models \
    --epochs 50

# 3. Use your trained models
model = DrugReflector(checkpoint_paths=[
    './my_models/model_fold_0.pt',
    './my_models/model_fold_1.pt',
    './my_models/model_fold_2.pt'
])
```

**See [drugreflector_training/README.md](drugreflector_training/README.md) for detailed training usage.**

## Quick Start

### For Inference Only

```bash
# 1. Install package
pip install drugreflector  # or clone and install locally

# 2. Download pre-trained models from Zenodo
# DOI: 10.5281/zenodo.16912445

# 3. Run inference
python drugreflector/predict.py input.h5ad \
    --model1 checkpoints/model_fold_0.pt \
    --model2 checkpoints/model_fold_1.pt \
    --model3 checkpoints/model_fold_2.pt
```

### For Training New Models

```bash
# 1. Install with training dependencies
pip install torch torchvision matplotlib tqdm scikit-learn

# 2. Prepare your LINCS data (see Training section)

# 3. Train models
python scripts/train.py --data-file training_data.pkl --output-dir ./models
```

## Installation

### Standard Installation (Inference Only)

```bash
# Clone repository
git clone https://github.com/Cellarity/drugreflector.git
cd drugreflector

# Install dependencies
pip install numpy pandas scipy torch anndata scanpy

# Download pre-trained models
# https://doi.org/10.5281/zenodo.16912445
```

### Installation with Training Support

```bash
# Additional dependencies for training
pip install matplotlib tqdm scikit-learn

# The training module is ready to use
python scripts/train.py --help
```

## Key Features

### Inference Module (Original)
- ✅ Load pre-trained 3-fold ensemble models
- ✅ Predict compound rankings from v-scores
- ✅ V-score computation from scRNA-seq data
- ✅ Signature refinement tools
- ✅ Command-line interface

### Training Module (New)
- ✅ Train custom DrugReflector models
- ✅ 3-fold ensemble training strategy
- ✅ Focal Loss for class imbalance
- ✅ Cosine Annealing learning rate schedule
- ✅ Compatible checkpoint format
- ✅ Training visualization and evaluation

## Model Architecture

Both pre-trained and newly trained models use the same `nnFC` architecture:

```
Input (978 landmark genes)
    ↓
FC(978 → 1024) + BatchNorm + ReLU + Dropout(0.64)
    ↓
FC(1024 → 2048) + BatchNorm + ReLU + Dropout(0.64)
    ↓
FC(2048 → 9597 compounds)
    ↓
Output (logits)
```

- **Parameters**: ~22.7M
- **Training**: Focal Loss (γ=2), Cosine Annealing LR
- **Ensemble**: Average of 3 models trained on different folds

## Training Data Requirements

To train your own models, you need:

1. **Expression Matrix**: 
   - Shape: (n_samples, 978)
   - 978 LINCS L1000 landmark genes
   - Log-transformed expression values

2. **Metadata**:
   - Compound IDs (`pert_id` column)
   - Sample identifiers
   - Optional: dose, time, cell line

3. **Format**:
   - Numpy array (.npy) or CSV for expression
   - CSV for metadata
   - OR: Pre-processed pickle file

See `scripts/prepare_data.py` for data preparation details.

## Example Workflows

### Workflow 1: Drug Screening with Pre-trained Models

```python
import scanpy as sc
from drugreflector import DrugReflector, compute_vscores_adata

# 1. Load your scRNA-seq data
adata = sc.read_h5ad('my_experiment.h5ad')

# 2. Compute v-scores between conditions
vscores = compute_vscores_adata(
    adata,
    group_col='treatment',
    group1_value='control',
    group2_value='disease'
)

# 3. Load pre-trained model
model = DrugReflector(checkpoint_paths=[
    'checkpoints/model_fold_0.pt',
    'checkpoints/model_fold_1.pt',
    'checkpoints/model_fold_2.pt'
])

# 4. Predict compounds
predictions = model.predict(vscores, n_top=100)
print(predictions.head(10))
```

### Workflow 2: Training Custom Models

```python
from drugreflector_training import DrugReflectorTrainer
import pickle

# 1. Load your training data
with open('my_training_data.pkl', 'rb') as f:
    data = pickle.load(f)

# 2. Train ensemble
trainer = DrugReflectorTrainer(
    device='cuda',
    num_epochs=50,
    batch_size=256
)

models, histories = trainer.train_ensemble(data, output_dir='./models')

# 3. Use trained models
from drugreflector import DrugReflector

model = DrugReflector(checkpoint_paths=[
    './models/model_fold_0.pt',
    './models/model_fold_1.pt',
    './models/model_fold_2.pt'
])
```

## Documentation

- **Inference**: See main [README.md](README.md)
- **Training**: See [drugreflector_training/README.md](drugreflector_training/README.md)
- **API Reference**: See docstrings in source code
- **Examples**: See `scripts/inference_example.py`

## Citation

```bibtex
@article{demeo2025drugreflector,
  title={Active learning framework leveraging transcriptomics identifies modulators of disease phenotypes},
  author={DeMeo, Benjamin and others},
  journal={Science},
  year={2025},
  doi={10.1126/science.adi8577}
}
```

**Model Checkpoints**: DOI 10.5281/zenodo.16912445

## Performance

Expected performance for trained models (Science 2025 SI Page 24):

| Dataset | Top 1% Recall |
|---------|---------------|
| CMap Touchstone | 0.46-0.50 |
| sciPlex3 | 0.30-0.35 |
| Internal | 0.60-0.65 |

Training time:
- Single model: 6-8 hours (RTX 3090)
- Full ensemble: 18-24 hours (RTX 3090)

## License

Copyright 2025, Cellarity Inc. See [LICENSE.txt](LICENSE.txt) for details.

**Commercial use requires prior written permission.**  
Contact: CellarityPublications@cellarity.com

## Support

- **Issues**: GitHub Issues
- **Documentation**: See README files
- **Training Questions**: See `drugreflector_training/README.md`
- **Inference Questions**: See main `README.md`

## Changelog

### Version 1.1.0 (New)
- ✨ Added training module
- ✨ Focal Loss implementation
- ✨ Training scripts and examples
- ✨ Compatible checkpoint format
- 📚 Extended documentation

### Version 1.0.0 (Original)
- 🎉 Initial release
- ✅ Inference with pre-trained models
- ✅ V-score computation
- ✅ Signature refinement