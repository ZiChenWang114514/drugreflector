# DrugReflector Training - Quick Reference

## 🚀 Quick Start (3 Commands)

```bash
# 1. Prepare data
python scripts/prepare_data.py \
    --expression-file data/expression.npy \
    --metadata-file data/metadata.csv \
    --output-file data/training_data.pkl

# 2. Train models
python scripts/train.py \
    --data-file data/training_data.pkl \
    --output-dir models

# 3. Use trained models
python scripts/inference_example.py --model-dir models
```

---

## 📁 File Structure

```
drugreflector_training/    # Training module
├── dataset.py             # PyTorch Dataset
├── losses.py              # Focal Loss
├── trainer.py             # Training logic
├── evaluator.py           # Evaluation
├── preprocessing.py       # Data prep
└── visualization.py       # Plots

scripts/                   # Executable scripts
├── prepare_data.py        # Data preparation
├── train.py               # Training script
└── inference_example.py   # Inference demo
```

---

## 🔧 Common Commands

### Train Full Ensemble (50 epochs)
```bash
python scripts/train.py \
    --data-file data/training_data.pkl \
    --output-dir models \
    --epochs 50
```

### Quick Test (Single Fold, 10 epochs)
```bash
python scripts/train.py \
    --data-file data/training_data.pkl \
    --output-dir test_models \
    --single-fold 0 \
    --epochs 10
```

### Custom Hyperparameters
```bash
python scripts/train.py \
    --data-file data/training_data.pkl \
    --output-dir custom_models \
    --epochs 100 \
    --batch-size 512 \
    --learning-rate 0.02 \
    --focal-gamma 2.5
```

### CPU Training (Slow)
```bash
python scripts/train.py \
    --data-file data/training_data.pkl \
    --output-dir models \
    --device cpu
```

---

## 📊 Data Format

### Input Files for prepare_data.py

**Expression Matrix** (`.npy` or `.csv`):
- Shape: `(n_samples, 978)`
- 978 LINCS L1000 landmark genes
- Values: Log-transformed expression

**Metadata** (`.csv`):
```csv
sample_id,pert_id,dose,time,cell_line
sample_001,BRD-K12345,10uM,24h,A549
sample_002,BRD-K12345,10uM,24h,A549
...
```

### Training Data Pickle

```python
{
    'X': np.ndarray,              # (n_samples, 978)
    'y': np.ndarray,              # (n_samples,) compound labels
    'folds': np.ndarray,          # (n_samples,) fold assignments
    'compound_names': list,       # Compound identifiers
    'gene_names': list,           # Gene names
    'metadata': pd.DataFrame      # Full metadata
}
```

---

## 🎯 Hyperparameter Reference

| Parameter | Default | Paper Source | Notes |
|-----------|---------|--------------|-------|
| `initial_lr` | 0.0139 | SI Table S5 | Initial learning rate |
| `min_lr` | 0.00001 | SI Page 3 | Minimum LR |
| `weight_decay` | 1e-5 | SI Table S5 | L2 regularization |
| `dropout` | 0.64 | SI Table S5 | Dropout rate |
| `focal_gamma` | 2.0 | SI Page 3 | Focal loss γ |
| `t_0` | 20 | SI Table S5 | Warm restart period |
| `batch_size` | 256 | - | Adjust for GPU |
| `epochs` | 50 | SI Page 3 | Total epochs |

---

## 🐍 Python API

### Train Programmatically

```python
from drugreflector_training import DrugReflectorTrainer
import pickle

# Load data
with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Train
trainer = DrugReflectorTrainer(
    device='cuda',
    num_epochs=50,
    batch_size=256
)

models, histories = trainer.train_ensemble(
    data, 
    output_dir='./models'
)
```

### Use Trained Models

```python
from drugreflector import DrugReflector
import pandas as pd

# Load models
model = DrugReflector(checkpoint_paths=[
    'models/model_fold_0.pt',
    'models/model_fold_1.pt',
    'models/model_fold_2.pt'
])

# Create v-scores
vscores = pd.Series(
    [1.2, -0.8, 0.5, ...],
    index=['TP53', 'EGFR', 'MYC', ...]
)

# Predict
predictions = model.predict(vscores, n_top=100)
```

### Evaluate Models

```python
from drugreflector_training import DrugReflectorEvaluator

evaluator = DrugReflectorEvaluator(models, device='cuda')

results = evaluator.evaluate(
    X_test, 
    y_test, 
    compound_names
)

print(f"Top 1% Recall: {results['top1_percent_recall']:.4f}")
```

---

## 📈 Expected Performance

| Dataset | Top 1% Recall | Training Time (RTX 3090) |
|---------|---------------|---------------------------|
| CMap Touchstone | 0.46-0.50 | ~18-24 hrs (3 models) |
| sciPlex3 | 0.30-0.35 | ~18-24 hrs (3 models) |
| Internal | 0.60-0.65 | ~18-24 hrs (3 models) |

---

## 🔍 Troubleshooting

### GPU Out of Memory
```bash
# Reduce batch size
python scripts/train.py --batch-size 128

# Or use low memory config
python scripts/train.py --batch-size 64 --device cuda
```

### Import Errors
```python
import sys
sys.path.append('/path/to/drugreflector')
```

### Slow Training
```python
# Use mixed precision (if PyTorch >= 1.6)
# Already enabled in trainer if GPU supports it

# Or reduce epochs for testing
python scripts/train.py --epochs 10 --single-fold 0
```

### Model Loading Errors
```python
# Verify checkpoint format
import torch
ckpt = torch.load('model_fold_0.pt')
print(ckpt.keys())

# Should have: model_state_dict, dimensions, params_init
```

---

## 📦 Output Files

After training, you'll have:

```
trained_models/
├── model_fold_0.pt              # Model checkpoint
├── model_fold_1.pt              # Model checkpoint
├── model_fold_2.pt              # Model checkpoint
├── ensemble_history.pkl         # Training history
├── training_history.png         # Training curves
├── fold_0_results.pkl           # Evaluation results
├── fold_1_results.pkl
└── fold_2_results.pkl
```

---

## 💡 Tips & Tricks

### Speed Up Training
- Use larger batch size (if GPU allows): `--batch-size 512`
- Reduce epochs for testing: `--epochs 10`
- Train single fold first: `--single-fold 0`

### Improve Performance
- More epochs: `--epochs 100`
- Adjust learning rate: `--learning-rate 0.01`
- Tune focal gamma: `--focal-gamma 2.5`

### Save GPU Memory
- Smaller batch: `--batch-size 64`
- Fewer workers: `--num-workers 2`
- Use CPU: `--device cpu` (very slow)

### Monitor Training
- Check `training_history.png`
- Look for steady recall improvement
- Verify learning rate schedule is working

---

## 📚 Documentation Links

- **Full Training Guide**: `drugreflector_training/README.md`
- **Refactoring Details**: `REFACTORING_SUMMARY.md`
- **Project Overview**: `PROJECT_README.md`
- **Original Inference**: `README.md`

---

## ✅ Validation Checklist

Before running full training:

- [ ] Data prepared with `prepare_data.py`
- [ ] Test run on single fold successful
- [ ] Checkpoint format validated
- [ ] Trained model loads with `DrugReflector`
- [ ] GPU memory sufficient for batch size
- [ ] Expected to run for ~24 hours

---

## 🎓 Citation

```bibtex
@article{demeo2025drugreflector,
  title={Active learning framework leveraging transcriptomics 
         identifies modulators of disease phenotypes},
  author={DeMeo, Benjamin and others},
  journal={Science},
  year={2025},
  doi={10.1126/science.adi8577}
}
```

---

## 🆘 Getting Help

1. Check documentation in `drugreflector_training/README.md`
2. Review examples in `scripts/`
3. Verify data format with `prepare_data.py --help`
4. Test with single fold and few epochs first

---

**Last Updated**: 2025-01
**Version**: 1.0.0