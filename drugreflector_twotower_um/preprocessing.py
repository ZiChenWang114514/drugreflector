"""
Data preprocessing utilities for Two-Tower DrugReflector training.

Implements the signature normalization described in Science 2025 SI Page 3.
"""
import numpy as np


def clip_and_normalize_signature(X: np.ndarray, clip_range=(-2, 2)) -> np.ndarray:
    """
    Clip and normalize signatures as described in the paper.
    
    From SI Page 3:
    "every transcriptional vector v is clipped to range [-2,2] such that 
    its standard deviation after clipping equals 1"
    
    Parameters
    ----------
    X : np.ndarray
        Raw expression matrix (n_samples, n_genes)
    clip_range : tuple, default=(-2, 2)
        Clipping range
    
    Returns
    -------
    np.ndarray
        Processed matrix with same shape as input
    """
    print(f"\n📊 Clipping and normalizing signatures...")
    print(f"   Clip range: {clip_range}")
    
    X_processed = np.zeros_like(X, dtype=np.float32)
    
    for i in range(X.shape[0]):
        # Step 1: Clip to specified range
        vec = np.clip(X[i], clip_range[0], clip_range[1])
        
        # Step 2: Normalize to std=1
        std = np.std(vec)
        if std > 0:
            vec = vec / std
        
        X_processed[i] = vec
    
    # Validation
    mean_std = np.std(X_processed, axis=1).mean()
    print(f"   ✓ Mean std after normalization: {mean_std:.4f}")
    print(f"   ✓ Data range: [{X_processed.min():.2f}, {X_processed.max():.2f}]")
    
    return X_processed


def normalize_mol_embeddings(mol_embeddings: np.ndarray, method: str = 'l2') -> np.ndarray:
    """
    Normalize molecular embeddings.
    
    Parameters
    ----------
    mol_embeddings : np.ndarray
        Molecular embeddings (n_samples, embedding_dim)
    method : str
        Normalization method: 'l2', 'zscore', or 'none'
    
    Returns
    -------
    np.ndarray
        Normalized embeddings
    """
    print(f"\n📊 Normalizing molecular embeddings (method: {method})...")
    
    if method == 'none':
        return mol_embeddings
    
    elif method == 'l2':
        norms = np.linalg.norm(mol_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        normalized = mol_embeddings / norms
        print(f"   ✓ L2 normalized, mean norm: {np.linalg.norm(normalized, axis=1).mean():.4f}")
        return normalized
    
    elif method == 'zscore':
        mean = mol_embeddings.mean(axis=0, keepdims=True)
        std = mol_embeddings.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-8)
        normalized = (mol_embeddings - mean) / std
        print(f"   ✓ Z-score normalized")
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_fold_splits(n_samples: int, n_folds: int = 3, random_state: int = 42) -> np.ndarray:
    """
    Create random fold assignments for cross-validation.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_folds : int, default=3
        Number of folds
    random_state : int, default=42
        Random seed
    
    Returns
    -------
    np.ndarray
        Fold assignments (n_samples,), values in [0, n_folds-1]
    """
    np.random.seed(random_state)
    folds = np.random.randint(0, n_folds, size=n_samples)
    
    print(f"\n📋 Created {n_folds}-fold splits:")
    for fold_id in range(n_folds):
        count = (folds == fold_id).sum()
        print(f"   Fold {fold_id}: {count:,} samples ({count/n_samples*100:.1f}%)")
    
    return folds
