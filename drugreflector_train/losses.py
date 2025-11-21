"""
Focal Loss implementation for DrugReflector training.

Reference: Lin et al. "Focal Loss for Dense Object Detection"
Paper: Science 2025 SI, Page 3 - "focal loss function with γ = 2"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Formula: FL(pt) = -(1 - pt)^γ * log(pt)
    
    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter (paper uses 2.0)
    alpha : Tensor or None, default=None
        Class weights
    reduction : str, default='mean'
        Reduction method: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Parameters
        ----------
        inputs : Tensor
            Model logits (N, C)
        targets : Tensor
            True labels (N,)
        
        Returns
        -------
        Tensor
            Focal loss value
        """
        # Cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute pt
        pt = torch.exp(-ce_loss)
        
        # Focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss