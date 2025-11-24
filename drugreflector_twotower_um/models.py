"""
Two-Tower Neural Network Models for DrugReflector Enhancement.

This module contains:
1. TranscriptomeTower: Encoder for 978-dim gene expression
2. MolecularTower: Encoder using Uni-Mol for molecular structures
3. TwoTowerModel: Combined model with fusion layer

Architecture based on DrugReflector paper enhancement proposal.
"""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def fetch_activation(activation_name: str, activation_init_params: Mapping[str, Any] | None = None) -> nn.Module:
    """Get activation function by name."""
    activation_init_params = {} if activation_init_params is None else activation_init_params
    activation_map = {
        'Sigmoid': nn.Sigmoid,
        'ReLU': nn.ReLU,
        'Mish': nn.Mish,
        'Tanh': nn.Tanh,
        'SELU': nn.SELU,
        'LeakyReLU': nn.LeakyReLU,
        'GELU': nn.GELU,
    }
    if activation_name not in activation_map:
        raise ValueError(f"Unknown activation: {activation_name}")
    return activation_map[activation_name](**activation_init_params)


def init_weights(m):
    """Initialize weights for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class TranscriptomeTower(nn.Module):
    """
    Transcriptome encoder tower.
    
    Takes 978-dim gene expression and outputs d-dim embedding.
    Architecture: input → 1024 → 2048 → embedding_dim
    
    Parameters
    ----------
    input_dim : int
        Number of input features (978 genes)
    embedding_dim : int
        Output embedding dimension
    hidden_dims : List[int]
        Hidden layer dimensions
    dropout_p : float
        Dropout probability
    activation : str
        Activation function name
    batch_norm : bool
        Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int = 978,
        embedding_dim: int = 512,
        hidden_dims: List[int] = [1024, 2048],
        dropout_p: float = 0.64,
        activation: str = 'ReLU',
        batch_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Build encoder layers
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(fetch_activation(activation))
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim
        
        # Final projection to embedding
        layers.append(nn.Linear(in_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        self.encoder.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Gene expression data, shape (batch_size, 978)
        
        Returns
        -------
        torch.Tensor
            Embedding, shape (batch_size, embedding_dim)
        """
        return self.encoder(x)


class MolecularTower(nn.Module):
    """
    Molecular structure encoder tower using Uni-Mol.
    
    Takes precomputed Uni-Mol embeddings and projects to d-dim embedding.
    
    Note: Uni-Mol embeddings should be precomputed during data preparation
    to avoid loading the large Uni-Mol model during training.
    
    Parameters
    ----------
    unimol_dim : int
        Dimension of Uni-Mol embeddings (512 for unimol_base)
    embedding_dim : int
        Output embedding dimension
    hidden_dims : List[int]
        Hidden layer dimensions for projection
    dropout_p : float
        Dropout probability
    activation : str
        Activation function name
    batch_norm : bool
        Whether to use batch normalization
    """
    
    def __init__(
        self,
        unimol_dim: int = 512,
        embedding_dim: int = 512,
        hidden_dims: List[int] = [1024],
        dropout_p: float = 0.3,
        activation: str = 'ReLU',
        batch_norm: bool = True,
    ):
        super().__init__()
        self.unimol_dim = unimol_dim
        self.embedding_dim = embedding_dim
        
        # Build projection layers
        layers = []
        in_dim = unimol_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(fetch_activation(activation))
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim
        
        # Final projection to embedding
        layers.append(nn.Linear(in_dim, embedding_dim))
        
        self.projector = nn.Sequential(*layers)
        self.projector.apply(init_weights)
    
    def forward(self, mol_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        mol_embedding : torch.Tensor
            Uni-Mol embeddings, shape (batch_size, unimol_dim)
        
        Returns
        -------
        torch.Tensor
            Projected embedding, shape (batch_size, embedding_dim)
        """
        return self.projector(mol_embedding)


class FusionLayer(nn.Module):
    """
    Fusion layer for combining transcriptome and molecular embeddings.
    
    Supports multiple fusion strategies:
    - 'concat': Concatenation followed by MLP
    - 'product': Element-wise product
    - 'attention': Cross-attention mechanism
    - 'gated': Gated fusion
    
    Parameters
    ----------
    embedding_dim : int
        Dimension of each tower's embedding
    fusion_type : str
        Fusion strategy ('concat', 'product', 'attention', 'gated')
    output_dim : int
        Output dimension after fusion
    dropout_p : float
        Dropout probability
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        fusion_type: str = 'concat',
        output_dim: int = 1024,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fusion_type = fusion_type
        self.output_dim = output_dim
        
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            )
        elif fusion_type == 'product':
            # Element-wise product followed by projection
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            )
        elif fusion_type == 'attention':
            self.query = nn.Linear(embedding_dim, embedding_dim)
            self.key = nn.Linear(embedding_dim, embedding_dim)
            self.value = nn.Linear(embedding_dim, embedding_dim)
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            )
        elif fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid()
            )
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.apply(init_weights)
    
    def forward(
        self, 
        h_transcript: torch.Tensor, 
        h_mol: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse transcriptome and molecular embeddings.
        
        Parameters
        ----------
        h_transcript : torch.Tensor
            Transcriptome embedding, shape (batch_size, embedding_dim)
        h_mol : torch.Tensor
            Molecular embedding, shape (batch_size, embedding_dim)
        
        Returns
        -------
        torch.Tensor
            Fused representation, shape (batch_size, output_dim)
        """
        if self.fusion_type == 'concat':
            h_combined = torch.cat([h_transcript, h_mol], dim=1)
            return self.fusion(h_combined)
        
        elif self.fusion_type == 'product':
            h_combined = h_transcript * h_mol
            return self.fusion(h_combined)
        
        elif self.fusion_type == 'attention':
            # Cross-attention: transcript attends to molecule
            q = self.query(h_transcript)
            k = self.key(h_mol)
            v = self.value(h_mol)
            
            attn_weights = F.softmax(
                torch.sum(q * k, dim=-1, keepdim=True) / np.sqrt(self.embedding_dim),
                dim=-1
            )
            h_attended = attn_weights * v
            h_combined = torch.cat([h_transcript, h_attended], dim=1)
            return self.fusion(h_combined)
        
        elif self.fusion_type == 'gated':
            h_cat = torch.cat([h_transcript, h_mol], dim=1)
            gate = self.gate(h_cat)
            h_gated = gate * h_transcript + (1 - gate) * h_mol
            return self.fusion(h_gated)


class TwoTowerModel(nn.Module):
    """
    Two-Tower Model for drug perturbation prediction.
    
    Architecture:
    - Tower 1: Transcriptome encoder (978 genes → embedding)
    - Tower 2: Molecular encoder (Uni-Mol features → embedding)
    - Fusion: Combine embeddings → classifier
    
    Parameters
    ----------
    n_genes : int
        Number of input genes (978)
    n_compounds : int
        Number of output compound classes
    embedding_dim : int
        Embedding dimension for each tower
    fusion_type : str
        Fusion strategy ('concat', 'product', 'attention', 'gated')
    transcript_hidden_dims : List[int]
        Hidden dims for transcriptome tower
    mol_hidden_dims : List[int]
        Hidden dims for molecular tower
    classifier_hidden_dims : List[int]
        Hidden dims for classifier
    transcript_dropout : float
        Dropout for transcriptome tower
    mol_dropout : float
        Dropout for molecular tower
    classifier_dropout : float
        Dropout for classifier
    unimol_dim : int
        Input dimension for Uni-Mol embeddings
    """
    
    def __init__(
        self,
        n_genes: int = 978,
        n_compounds: int = 10000,
        embedding_dim: int = 512,
        fusion_type: str = 'concat',
        transcript_hidden_dims: List[int] = [1024, 2048],
        mol_hidden_dims: List[int] = [1024],
        classifier_hidden_dims: List[int] = [2048, 1024],
        transcript_dropout: float = 0.64,
        mol_dropout: float = 0.3,
        classifier_dropout: float = 0.3,
        unimol_dim: int = 512,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_compounds = n_compounds
        self.embedding_dim = embedding_dim
        self.fusion_type = fusion_type
        
        # Tower 1: Transcriptome
        self.transcript_tower = TranscriptomeTower(
            input_dim=n_genes,
            embedding_dim=embedding_dim,
            hidden_dims=transcript_hidden_dims,
            dropout_p=transcript_dropout,
        )
        
        # Tower 2: Molecular
        self.mol_tower = MolecularTower(
            unimol_dim=unimol_dim,
            embedding_dim=embedding_dim,
            hidden_dims=mol_hidden_dims,
            dropout_p=mol_dropout,
        )
        
        # Fusion layer
        fusion_output_dim = classifier_hidden_dims[0] if classifier_hidden_dims else embedding_dim
        self.fusion = FusionLayer(
            embedding_dim=embedding_dim,
            fusion_type=fusion_type,
            output_dim=fusion_output_dim,
            dropout_p=classifier_dropout,
        )
        
        # Classifier
        classifier_layers = []
        in_dim = fusion_output_dim
        
        for i, h_dim in enumerate(classifier_hidden_dims[1:] if classifier_hidden_dims else []):
            classifier_layers.append(nn.Linear(in_dim, h_dim))
            classifier_layers.append(nn.BatchNorm1d(h_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(classifier_dropout))
            in_dim = h_dim
        
        classifier_layers.append(nn.Linear(in_dim, n_compounds))
        
        self.classifier = nn.Sequential(*classifier_layers)
        self.classifier.apply(init_weights)
    
    def forward(
        self, 
        gene_expr: torch.Tensor, 
        mol_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        gene_expr : torch.Tensor
            Gene expression data, shape (batch_size, n_genes)
        mol_embedding : torch.Tensor
            Uni-Mol embeddings, shape (batch_size, unimol_dim)
        
        Returns
        -------
        torch.Tensor
            Logits, shape (batch_size, n_compounds)
        """
        # Encode both modalities
        h_transcript = self.transcript_tower(gene_expr)
        h_mol = self.mol_tower(mol_embedding)
        
        # Fuse
        h_fused = self.fusion(h_transcript, h_mol)
        
        # Classify
        logits = self.classifier(h_fused)
        
        return logits
    
    def get_embeddings(
        self, 
        gene_expr: torch.Tensor, 
        mol_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get intermediate embeddings for analysis.
        
        Returns transcriptome embedding, molecular embedding, and fused embedding.
        """
        h_transcript = self.transcript_tower(gene_expr)
        h_mol = self.mol_tower(mol_embedding)
        h_fused = self.fusion(h_transcript, h_mol)
        return h_transcript, h_mol, h_fused


class TranscriptOnlyModel(nn.Module):
    """
    Baseline model using only transcriptome data (original DrugReflector).
    
    This is for comparison with the two-tower model.
    """
    
    def __init__(
        self,
        n_genes: int = 978,
        n_compounds: int = 10000,
        hidden_dims: List[int] = [1024, 2048],
        dropout_p: float = 0.64,
        activation: str = 'ReLU',
        batch_norm: bool = True,
    ):
        super().__init__()
        
        layers = []
        in_dim = n_genes
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(fetch_activation(activation))
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, n_compounds))
        
        self.network = nn.Sequential(*layers)
        self.network.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
