"""
Data preprocessing utilities for DrugReflector training.

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
    
    X_processed = np.zeros_like(X)
    
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