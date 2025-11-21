"""
PyTorch Dataset for LINCS training data.
"""
import sys
sys.path.append('.')

import numpy as np
import torch
from torch.utils.data import Dataset


class LINCSDataset(Dataset):
    """
    LINCS dataset for DrugReflector training.
    
    Parameters
    ----------
    X : np.ndarray
        Expression matrix (n_samples, 978 genes)
    y : np.ndarray
        Compound labels (n_samples,)
    fold_mask : np.ndarray, optional
        Boolean mask indicating which samples to include
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, fold_mask: np.ndarray = None):
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