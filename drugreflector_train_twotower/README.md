# Two-Tower DrugReflector Training

Enhanced DrugReflector with molecular structure integration using Uni-Mol.

## Architecture

```
Tower 1 (Transcriptome):
  Input: 978 genes → [1024] → [2048] → 512-dim embedding

Tower 2 (Molecular):
  Input: Uni-Mol features (512-dim) → [1024] → 512-dim embedding

Fusion:
  [h_transcript, h_mol] → concat/product/attention/gated → classifier → compounds
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Install Uni-Mol
pip install unimol-tools
# Or from source:
# git clone https://github.com/dptech-corp/Uni-Mol.git
# cd Uni-Mol/unimol_tools && pip install .
```

### 2. Precompute Molecular Embeddings (One-time)

```bash
python precompute_mol_embeddings.py \
    --compound-info ../datasets/LINCS2020/compoundinfo_beta.txt \
    --training-data ../datasets/training_data_lincs2020_final.pkl \
    --output ../datasets/mol_embeddings_unimol.pkl \
    --batch-size 64
```

### 3. Train Model

```bash
# Train all folds (ensemble)
python train.py \
    --data-file ../datasets/training_data_lincs2020_final.pkl \
    --mol-embeddings ../datasets/mol_embeddings_unimol.pkl \
    --output-dir ../models/twotower \
    --all-folds

# Or train single fold
python train.py \
    --data-file ../datasets/training_data_lincs2020_final.pkl \
    --mol-embeddings ../datasets/mol_embeddings_unimol.pkl \
    --output-dir ../models/twotower \
    --fold 0
```

### 4. Evaluate

```bash
python eval.py \
    --model-path ../models/twotower/twotower_fold_0.pt \
    --data-file ../datasets/training_data_lincs2020_final.pkl \
    --mol-embeddings ../datasets/mol_embeddings_unimol.pkl \
    --fold 0
```

## HPC Usage

```bash
# Submit precompute job
sbatch scripts/precompute_embeddings.sh

# Submit training job
sbatch scripts/train_ensemble.sh
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embedding-dim` | 512 | Embedding dimension for each tower |
| `--fusion-type` | concat | Fusion strategy: concat, product, attention, gated |
| `--contrastive-weight` | 0.1 | Weight for cross-modal contrastive loss |
| `--focal-gamma` | 2.0 | Focal loss gamma |
| `--learning-rate` | 0.0139 | Initial learning rate |
| `--epochs` | 50 | Number of training epochs |

## File Structure

```
drugreflector_train_twotower/
├── __init__.py
├── models.py              # TwoTowerModel, TranscriptomeTower, MolecularTower
├── unimol_encoder.py      # Uni-Mol molecular encoder
├── dataset.py             # TwoTowerDataset
├── trainer.py             # TwoTowerTrainer
├── losses.py              # FocalLoss, ContrastiveLoss, TwoTowerLoss
├── preprocessing.py       # Data normalization
├── train.py               # Main training script
├── eval.py                # Evaluation script
├── precompute_mol_embeddings.py
├── requirements.txt
└── scripts/
    ├── precompute_embeddings.sh
    ├── train_ensemble.sh
    └── train_single_fold.sh
```
