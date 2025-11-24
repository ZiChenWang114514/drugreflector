"""
Two-Tower DrugReflector Training Package

Enhanced DrugReflector with molecular structure integration using Uni-Mol.

Architecture:
- Tower 1: Transcriptome encoder (978 genes → embedding)
- Tower 2: Molecular encoder (Uni-Mol features → embedding)
- Fusion: Combined representation → classifier

Usage:
    # Step 1: Precompute molecular embeddings
    python precompute_mol_embeddings.py --compound-info data/compoundinfo_beta.txt --output embeddings.pkl
    
    # Step 2: Train model
    python train.py --data-file data.pkl --mol-embeddings embeddings.pkl --output-dir models/ --all-folds
"""

from .models import TwoTowerModel, TranscriptomeTower, MolecularTower, FusionLayer
from .trainer import TwoTowerTrainer
from .dataset import TwoTowerDataset, load_training_data_with_mol
from .losses import FocalLoss, TwoTowerLoss, ContrastiveLoss
from .preprocessing import clip_and_normalize_signature, normalize_mol_embeddings

__version__ = "1.0.0"
__all__ = [
    "TwoTowerModel",
    "TranscriptomeTower", 
    "MolecularTower",
    "FusionLayer",
    "TwoTowerTrainer",
    "TwoTowerDataset",
    "load_training_data_with_mol",
    "FocalLoss",
    "TwoTowerLoss",
    "ContrastiveLoss",
    "clip_and_normalize_signature",
    "normalize_mol_embeddings",
]
