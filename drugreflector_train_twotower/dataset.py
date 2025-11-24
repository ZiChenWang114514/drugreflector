"""
PyTorch Dataset for Two-Tower DrugReflector Training.

Handles both transcriptome data and molecular embeddings.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    """
    Dataset for Two-Tower DrugReflector training.
    
    Provides both gene expression data and molecular embeddings.
    
    Parameters
    ----------
    X : np.ndarray
        Gene expression matrix (n_samples, 978 genes)
    y : np.ndarray
        Compound labels (n_samples,)
    mol_embeddings : np.ndarray
        Molecular embeddings for each sample (n_samples, mol_dim)
    fold_mask : np.ndarray, optional
        Boolean mask indicating which samples to include
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        mol_embeddings: np.ndarray,
        fold_mask: np.ndarray = None,
    ):
        if fold_mask is not None:
            self.X = torch.FloatTensor(X[fold_mask])
            self.y = torch.LongTensor(y[fold_mask])
            self.mol_embeddings = torch.FloatTensor(mol_embeddings[fold_mask])
        else:
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
            self.mol_embeddings = torch.FloatTensor(mol_embeddings)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.mol_embeddings[idx], self.y[idx]


class TranscriptOnlyDataset(Dataset):
    """
    Dataset for transcript-only baseline (original DrugReflector).
    
    Parameters
    ----------
    X : np.ndarray
        Gene expression matrix (n_samples, 978 genes)
    y : np.ndarray
        Compound labels (n_samples,)
    fold_mask : np.ndarray, optional
        Boolean mask indicating which samples to include
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        fold_mask: np.ndarray = None,
    ):
        if fold_mask is not None:
            self.X = torch.FloatTensor(X[fold_mask])
            self.y = torch.LongTensor(y[fold_mask])
        else:
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_molecular_embeddings(
    training_data: Dict,
    pert_to_embedding: Dict[str, np.ndarray],
    fallback_embedding: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare molecular embeddings aligned with training data samples.
    
    Parameters
    ----------
    training_data : Dict
        Training data dictionary with 'sample_meta' containing 'pert_id'
    pert_to_embedding : Dict[str, np.ndarray]
        Dictionary mapping pert_id to molecular embedding
    fallback_embedding : np.ndarray, optional
        Fallback embedding for missing compounds (zeros if not provided)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - mol_embeddings: Shape (n_samples, embedding_dim)
        - valid_mask: Boolean array indicating samples with valid embeddings
    """
    sample_meta = training_data['sample_meta']
    n_samples = len(sample_meta)
    
    # Get embedding dimension
    sample_embedding = next(iter(pert_to_embedding.values()))
    embedding_dim = len(sample_embedding)
    
    print(f"\n📊 Preparing molecular embeddings...")
    print(f"  Samples: {n_samples:,}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Available compounds: {len(pert_to_embedding):,}")
    
    # Initialize arrays
    mol_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)
    valid_mask = np.zeros(n_samples, dtype=bool)
    
    if fallback_embedding is None:
        fallback_embedding = np.zeros(embedding_dim, dtype=np.float32)
    
    # Map embeddings to samples
    missing_count = 0
    pert_ids = sample_meta['pert_id'].values
    
    for i, pert_id in enumerate(pert_ids):
        if pert_id in pert_to_embedding:
            mol_embeddings[i] = pert_to_embedding[pert_id]
            valid_mask[i] = True
        else:
            mol_embeddings[i] = fallback_embedding
            missing_count += 1
    
    n_valid = valid_mask.sum()
    print(f"  ✓ Samples with embeddings: {n_valid:,} ({n_valid/n_samples*100:.1f}%)")
    if missing_count > 0:
        print(f"  ⚠️ Samples missing embeddings: {missing_count:,}")
    
    return mol_embeddings, valid_mask


def load_training_data_with_mol(
    training_data_path: str,
    mol_embeddings_path: str,
    compound_info_path: Optional[str] = None,
    filter_missing_mol: bool = True,
) -> Dict:
    """
    Load training data and merge with molecular embeddings.
    
    Parameters
    ----------
    training_data_path : str
        Path to training data pickle
    mol_embeddings_path : str
        Path to molecular embeddings pickle
    compound_info_path : str, optional
        Path to compoundinfo_beta.txt (for computing embeddings on-the-fly if needed)
    filter_missing_mol : bool
        Whether to filter out samples without molecular embeddings
    
    Returns
    -------
    Dict
        Enhanced training data with molecular embeddings
    """
    print(f"\n{'='*80}")
    print(f"📂 Loading Training Data with Molecular Embeddings")
    print(f"{'='*80}")
    
    # Load training data
    print(f"\n📖 Loading training data from: {training_data_path}")
    with open(training_data_path, 'rb') as f:
        training_data = pickle.load(f)
    
    print(f"  ✓ Samples: {len(training_data['X']):,}")
    print(f"  ✓ Compounds: {len(training_data['compound_names']):,}")
    
    # Load molecular embeddings
    print(f"\n📖 Loading molecular embeddings from: {mol_embeddings_path}")
    with open(mol_embeddings_path, 'rb') as f:
        pert_to_embedding = pickle.load(f)
    
    print(f"  ✓ Compounds with embeddings: {len(pert_to_embedding):,}")
    
    # Prepare sample-level embeddings
    mol_embeddings, valid_mask = prepare_molecular_embeddings(
        training_data=training_data,
        pert_to_embedding=pert_to_embedding,
    )
    
    # Filter if requested
    if filter_missing_mol and not valid_mask.all():
        print(f"\n⚠️ Filtering samples without molecular embeddings...")
        
        n_before = len(training_data['X'])
        
        training_data['X'] = training_data['X'][valid_mask]
        training_data['y'] = training_data['y'][valid_mask]
        training_data['folds'] = training_data['folds'][valid_mask]
        training_data['sample_meta'] = training_data['sample_meta'][valid_mask].reset_index(drop=True)
        mol_embeddings = mol_embeddings[valid_mask]
        
        # Rebuild compound mapping if needed
        remaining_compounds = set(training_data['sample_meta']['pert_id'].unique())
        training_data['compound_names'] = sorted(list(remaining_compounds))
        training_data['pert_to_idx'] = {p: i for i, p in enumerate(training_data['compound_names'])}
        
        # Update labels
        training_data['y'] = np.array([
            training_data['pert_to_idx'][p] 
            for p in training_data['sample_meta']['pert_id']
        ], dtype=np.int32)
        
        n_after = len(training_data['X'])
        print(f"  Filtered: {n_before:,} → {n_after:,} samples")
        print(f"  Remaining compounds: {len(training_data['compound_names']):,}")
    
    # Add molecular embeddings to training data
    training_data['mol_embeddings'] = mol_embeddings
    training_data['pert_to_mol_embedding'] = pert_to_embedding
    
    print(f"\n✅ Data loaded successfully!")
    print(f"  Final samples: {len(training_data['X']):,}")
    print(f"  Final compounds: {len(training_data['compound_names']):,}")
    print(f"  Molecular embedding dim: {mol_embeddings.shape[1]}")
    
    return training_data


def create_dataloaders(
    training_data: Dict,
    fold_id: int,
    batch_size: int = 256,
    num_workers: int = 4,
    use_two_tower: bool = True,
) -> Tuple:
    """
    Create train and validation DataLoaders.
    
    Parameters
    ----------
    training_data : Dict
        Training data dictionary
    fold_id : int
        Which fold to use as validation
    batch_size : int
        Batch size
    num_workers : int
        Number of DataLoader workers
    use_two_tower : bool
        Whether to use TwoTowerDataset or TranscriptOnlyDataset
    
    Returns
    -------
    Tuple
        (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    X = training_data['X']
    y = training_data['y']
    folds = training_data['folds']
    
    val_mask = folds == fold_id
    train_mask = ~val_mask
    
    if use_two_tower:
        mol_embeddings = training_data['mol_embeddings']
        
        train_dataset = TwoTowerDataset(X, y, mol_embeddings, train_mask)
        val_dataset = TwoTowerDataset(X, y, mol_embeddings, val_mask)
    else:
        train_dataset = TranscriptOnlyDataset(X, y, train_mask)
        val_dataset = TranscriptOnlyDataset(X, y, val_mask)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
