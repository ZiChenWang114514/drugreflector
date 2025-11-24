"""
Loss functions for Two-Tower DrugReflector training.

Includes:
- FocalLoss: Addressing class imbalance (from original DrugReflector)
- ContrastiveLoss: For cross-modal learning
- CombinedLoss: Multi-task loss combination

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
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for cross-modal alignment.
    
    Encourages transcriptome and molecular embeddings of the same
    compound to be close, and different compounds to be far.
    
    Parameters
    ----------
    temperature : float
        Temperature scaling factor
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        h_transcript: torch.Tensor, 
        h_mol: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Parameters
        ----------
        h_transcript : torch.Tensor
            Transcriptome embeddings (batch_size, embedding_dim)
        h_mol : torch.Tensor
            Molecular embeddings (batch_size, embedding_dim)
        labels : torch.Tensor
            Compound labels (batch_size,)
        
        Returns
        -------
        torch.Tensor
            Contrastive loss
        """
        # Normalize embeddings
        h_transcript = F.normalize(h_transcript, p=2, dim=1)
        h_mol = F.normalize(h_mol, p=2, dim=1)
        
        # Compute similarity matrix
        # Each transcriptome embedding should match its corresponding molecular embedding
        logits = torch.matmul(h_transcript, h_mol.t()) / self.temperature
        
        # Labels are just indices (diagonal should be positive)
        batch_size = h_transcript.size(0)
        targets = torch.arange(batch_size, device=h_transcript.device)
        
        # Cross-entropy in both directions
        loss_t2m = F.cross_entropy(logits, targets)
        loss_m2t = F.cross_entropy(logits.t(), targets)
        
        return (loss_t2m + loss_m2t) / 2


class TwoTowerLoss(nn.Module):
    """
    Combined loss for Two-Tower model training.
    
    Combines:
    1. Classification loss (Focal Loss)
    2. Cross-modal contrastive loss (optional)
    
    Parameters
    ----------
    focal_gamma : float
        Focal loss gamma
    contrastive_weight : float
        Weight for contrastive loss (0 to disable)
    contrastive_temp : float
        Temperature for contrastive loss
    """
    
    def __init__(
        self, 
        focal_gamma: float = 2.0,
        contrastive_weight: float = 0.1,
        contrastive_temp: float = 0.07,
    ):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.contrastive_weight = contrastive_weight
        
        if contrastive_weight > 0:
            self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temp)
        else:
            self.contrastive_loss = None
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        h_transcript: torch.Tensor = None,
        h_mol: torch.Tensor = None,
    ) -> dict:
        """
        Compute combined loss.
        
        Parameters
        ----------
        logits : torch.Tensor
            Classification logits
        targets : torch.Tensor
            Target labels
        h_transcript : torch.Tensor, optional
            Transcriptome embeddings (for contrastive loss)
        h_mol : torch.Tensor, optional
            Molecular embeddings (for contrastive loss)
        
        Returns
        -------
        dict
            Dictionary with 'total', 'focal', and 'contrastive' losses
        """
        # Classification loss
        focal = self.focal_loss(logits, targets)
        
        # Contrastive loss
        if self.contrastive_loss is not None and h_transcript is not None and h_mol is not None:
            contrastive = self.contrastive_loss(h_transcript, h_mol, targets)
            total = focal + self.contrastive_weight * contrastive
        else:
            contrastive = torch.tensor(0.0, device=logits.device)
            total = focal
        
        return {
            'total': total,
            'focal': focal,
            'contrastive': contrastive,
        }


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Parameters
    ----------
    smoothing : float
        Label smoothing factor (0-1)
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        
        # Create smooth labels
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        
        return loss
