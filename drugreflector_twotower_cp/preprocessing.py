"""
Data preprocessing utilities.
"""
import numpy as np


def clip_and_normalize_signature(X: np.ndarray, clip_range=(-2, 2)) -> np.ndarray:
    """
    Clip and normalize signatures.
    
    From SI Page 3: clip to [-2,2] such that std=1 after clipping.
    """
    print(f"\n📊 Preprocessing signatures...")
    
    X_processed = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        vec = np.clip(X[i], clip_range[0], clip_range[1])
        std = np.std(vec)
        if std > 0:
            vec = vec / std
        X_processed[i] = vec
    
    print(f"  ✓ Mean std: {np.std(X_processed, axis=1).mean():.4f}")
    
    return X_processed