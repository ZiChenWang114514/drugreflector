"""
Two-Tower Model Architecture

Integrates transcriptome (gene expression) and chemical structure (molecular graph)
information using a dual-encoder architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

try:
    from chemprop import nn as cnn
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False
    print("⚠️  Chemprop not available. Install with: pip install chemprop")


class ChemicalEncoder(nn.Module):
    """
    Chemical structure encoder using Chemprop2 MPNN with optional 3D coordinates.
    
    Integrates 2D graph topology with 3D geometric information for enhanced
    molecular representation learning.
    
    Parameters
    ----------
    output_dim : int
        Output embedding dimension
    mpnn_depth : int
        Message passing depth
    dropout : float
        Dropout probability
    use_3d : bool
        Whether to incorporate 3D coordinates
    d_coord : int
        Dimension of 3D coordinate features (xyz + distances)
    """
    
    def __init__(
        self, 
        output_dim: int = 512, 
        mpnn_depth: int = 3, 
        dropout: float = 0.0,
        use_3d: bool = False,
        d_coord: int = 16
    ):
        super().__init__()
        
        if not CHEMPROP_AVAILABLE:
            raise ImportError("Chemprop2 is required for chemical encoder")
        
        self.output_dim = output_dim
        self.use_3d = use_3d
        self.d_coord = d_coord
        
        # Determine atom feature dimension
        # Default: 133 (Chemprop2 v2 atom featurizer)
        # With 3D: 133 + d_coord (coordinate features)
        d_v_base = 133
        d_v = d_v_base + d_coord if use_3d else d_v_base
        
        # Chemprop2 MPNN with expanded atom features
        self.mpnn = cnn.BondMessagePassing(
            d_v=d_v,
            d_e=14,  # Default bond feature dim
            d_h=output_dim,
            depth=mpnn_depth,
            dropout=dropout,
            activation='relu'
        )
        
        # 3D coordinate encoder (if enabled)
        if use_3d:
            self.coord_encoder = nn.Sequential(
                nn.Linear(3, d_coord // 2),  # xyz -> hidden
                nn.ReLU(),
                nn.Linear(d_coord // 2, d_coord)  # hidden -> d_coord
            )
        
        # Aggregation
        self.agg = cnn.MeanAggregation()
        
        # Batch norm
        self.bn = nn.BatchNorm1d(output_dim)
    
    def _compute_3d_features(self, coords):
        """
        Compute 3D geometric features from atomic coordinates.
        
        Features include:
        - Encoded XYZ coordinates
        - Distance to centroid
        - Distance statistics
        
        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates (n_atoms, 3)
        
        Returns
        -------
        torch.Tensor
            3D features (n_atoms, d_coord)
        """
        # Encode raw coordinates
        coord_encoded = self.coord_encoder(coords)  # (n_atoms, d_coord)
        
        # Compute centroid
        centroid = coords.mean(dim=0, keepdim=True)  # (1, 3)
        
        # Distance to centroid (geometric center)
        dist_to_center = torch.norm(coords - centroid, dim=1, keepdim=True)  # (n_atoms, 1)
        
        # Normalize distance
        dist_normalized = dist_to_center / (dist_to_center.max() + 1e-8)
        
        # Combine encoded coords with distance info
        # Simply concatenate and project back to d_coord
        combined = torch.cat([coord_encoded, dist_normalized], dim=1)
        
        # Project to target dimension
        if combined.shape[1] != self.d_coord:
            proj = nn.Linear(combined.shape[1], self.d_coord).to(coords.device)
            combined = proj(combined)
        
        return combined[:, :self.d_coord]  # Ensure exact dimension
    
    def forward(self, bmg, coords_3d=None):
        """
        Forward pass with optional 3D coordinates.
        
        Parameters
        ----------
        bmg : BatchMolGraph
            Batch of molecular graphs from Chemprop
        coords_3d : torch.Tensor, optional
            3D coordinates (n_atoms_total, 3) for all atoms in batch
            If provided and use_3d=True, will be integrated into atom features
        
        Returns
        -------
        torch.Tensor
            Molecular embeddings (batch_size, output_dim)
        """
        # 🔥 关键修复：检查 bmg 是否有效
        if bmg is None or not hasattr(bmg, 'V') or bmg.V is None:
            raise ValueError(
                "Invalid BatchMolGraph: bmg.V is None. "
                "Check that molecules were properly featurized in collate_fn."
            )
        
        # Integrate 3D coordinates if available
        if self.use_3d and coords_3d is not None:
            # Compute 3D features
            coord_features = self._compute_3d_features(coords_3d)  # (n_atoms, d_coord)
            
            # 扩展原子特征（如果维度匹配）
            if coord_features.shape[0] == bmg.V.shape[0]:
                bmg.V = torch.cat([bmg.V, coord_features], dim=1)
        
        # Message passing
        H_v = self.mpnn(bmg)
        
        # Aggregate to molecule level
        h_mol = self.agg(H_v, bmg.batch)
        
        # Normalize
        h_mol = self.bn(h_mol)
        
        return h_mol


class TranscriptomeEncoder(nn.Module):
    """
    Transcriptome (gene expression) encoder using MLP.
    
    Parameters
    ----------
    input_dim : int
        Number of genes (978 for landmark genes)
    hidden_dims : List[int]
        Hidden layer dimensions
    dropout_p : float
        Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [1024, 2048],
        dropout_p: float = 0.64
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.BatchNorm1d(h_dim)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Gene expression (batch_size, n_genes)
        
        Returns
        -------
        torch.Tensor
            Transcriptome embeddings (batch_size, output_dim)
        """
        return self.encoder(x)


class FusionModule(nn.Module):
    """
    Feature fusion module for combining modalities.
    
    Parameters
    ----------
    chem_dim : int
        Chemical embedding dimension
    transcript_dim : int
        Transcriptome embedding dimension
    method : str
        Fusion method: 'concat', 'multiply', 'add', 'gated'
    """
    
    def __init__(
        self,
        chem_dim: int,
        transcript_dim: int,
        method: str = 'concat'
    ):
        super().__init__()
        self.method = method
        self.chem_dim = chem_dim
        self.transcript_dim = transcript_dim
        
        if method == 'concat':
            self.output_dim = chem_dim + transcript_dim
        elif method == 'multiply' or method == 'add':
            if chem_dim != transcript_dim:
                # Project to same dimension
                self.chem_proj = nn.Linear(chem_dim, transcript_dim)
                self.output_dim = transcript_dim
            else:
                self.chem_proj = None
                self.output_dim = chem_dim
        elif method == 'gated':
            # Gated fusion
            if chem_dim != transcript_dim:
                self.chem_proj = nn.Linear(chem_dim, transcript_dim)
                hidden_dim = transcript_dim
            else:
                self.chem_proj = None
                hidden_dim = chem_dim
            
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.output_dim = hidden_dim
        else:
            raise ValueError(f"Unknown fusion method: {method}")
    
    def forward(self, h_chem, h_transcript):
        """
        Fuse chemical and transcriptome embeddings.
        
        Parameters
        ----------
        h_chem : torch.Tensor
            Chemical embeddings
        h_transcript : torch.Tensor
            Transcriptome embeddings
        
        Returns
        -------
        torch.Tensor
            Fused embeddings
        """
        if self.method == 'concat':
            return torch.cat([h_chem, h_transcript], dim=1)
        
        elif self.method == 'multiply':
            if self.chem_proj is not None:
                h_chem = self.chem_proj(h_chem)
            return h_chem * h_transcript
        
        elif self.method == 'add':
            if self.chem_proj is not None:
                h_chem = self.chem_proj(h_chem)
            return h_chem + h_transcript
        
        elif self.method == 'gated':
            if self.chem_proj is not None:
                h_chem = self.chem_proj(h_chem)
            
            # Gate weights
            gate_input = torch.cat([h_chem, h_transcript], dim=1)
            alpha = self.gate(gate_input)
            
            # Gated combination
            return alpha * h_chem + (1 - alpha) * h_transcript


class TwoTowerModel(nn.Module):
    """
    Two-Tower DrugReflector Model.
    
    Combines transcriptome and chemical structure information.
    
    Parameters
    ----------
    n_genes : int
        Number of gene features
    n_compounds : int
        Number of compound classes
    chem_hidden_dim : int
        Chemical encoder output dimension
    transcript_hidden_dims : List[int]
        Transcriptome encoder hidden dimensions
    fusion_method : str
        Feature fusion method
    mpnn_depth : int
        MPNN depth
    mpnn_dropout : float
        MPNN dropout
    """
    
    def __init__(
        self,
        n_genes: int,
        n_compounds: int,
        chem_hidden_dim: int = 512,
        transcript_hidden_dims: List[int] = [1024, 2048],
        fusion_method: str = 'concat',
        mpnn_depth: int = 3,
        mpnn_dropout: float = 0.0,
        use_3d: bool = False,
        d_coord: int = 16,
        conformer_method: str = 'ETKDG'
    ):
        super().__init__()
        
        # Chemical encoder (Tower 1)
        self.chem_encoder = ChemicalEncoder(
            output_dim=chem_hidden_dim,
            mpnn_depth=mpnn_depth,
            dropout=mpnn_dropout,
            use_3d=use_3d,
            d_coord=d_coord,
            # conformer_method=conformer_method
        )
        
        # Transcriptome encoder (Tower 2)
        self.transcript_encoder = TranscriptomeEncoder(
            input_dim=n_genes,
            hidden_dims=transcript_hidden_dims,
            dropout_p=0.64
        )
        
        # Fusion module
        self.fusion = FusionModule(
            chem_dim=chem_hidden_dim,
            transcript_dim=transcript_hidden_dims[-1],
            method=fusion_method
        )
        
        # Classifier head
        self.classifier = nn.Linear(self.fusion.output_dim, n_compounds)
        
        # Apply weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_transcript, bmg_chem):
        """
        Forward pass.
        
        Parameters
        ----------
        x_transcript : torch.Tensor
            Gene expression (batch_size, n_genes)
        bmg_chem : BatchMolGraph
            Batch of molecular graphs
        
        Returns
        -------
        torch.Tensor
            Logits (batch_size, n_compounds)
        """
        # Encode both modalities
        h_chem = self.chem_encoder(bmg_chem)
        h_transcript = self.transcript_encoder(x_transcript)
        
        # Fuse features
        h_fused = self.fusion(h_chem, h_transcript)
        
        # Classify
        logits = self.classifier(h_fused)
        
        return logits
    
    def get_embeddings(self, x_transcript, bmg_chem):
        """
        Get intermediate embeddings (for analysis).
        
        Returns
        -------
        dict
            Dictionary containing embeddings from each module
        """
        h_chem = self.chem_encoder(bmg_chem)
        h_transcript = self.transcript_encoder(x_transcript)
        h_fused = self.fusion(h_chem, h_transcript)
        
        return {
            'chemical': h_chem,
            'transcriptome': h_transcript,
            'fused': h_fused
        }