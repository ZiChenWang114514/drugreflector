"""
PyTorch Dataset for Two-Tower training.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional

try:
    from chemprop.data import MoleculeDatapoint, MoleculeDataset
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False


class TwoTowerDataset(Dataset):
    """
    Dataset combining transcriptome and chemical structure.
    
    Parameters
    ----------
    X : np.ndarray
        Gene expression matrix (n_samples, n_genes)
    y : np.ndarray
        Compound labels (n_samples,)
    compound_ids : np.ndarray
        Compound IDs for each sample (n_samples,)
    smiles_dict : Dict[str, str]
        Mapping from compound ID to SMILES
    fold_mask : np.ndarray, optional
        Boolean mask for samples to include
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        compound_ids: np.ndarray,
        smiles_dict: Dict[str, str],
        fold_mask: Optional[np.ndarray] = None
    ):
        if not CHEMPROP_AVAILABLE:
            raise ImportError("Chemprop is required")
        
        # Apply fold mask
        if fold_mask is not None:
            self.X = X[fold_mask]
            self.y = y[fold_mask]
            self.compound_ids = compound_ids[fold_mask]
        else:
            self.X = X
            self.y = y
            self.compound_ids = compound_ids
        
        self.smiles_dict = smiles_dict
        
        # Create molecule featurizer
        self.mol_featurizer = SimpleMoleculeMolGraphFeaturizer()
        
        # Pre-featurize molecules and cache
        print("  🔬 Pre-featurizing molecules...")
        self.mol_cache = {}
        unique_compounds = np.unique(self.compound_ids)
        
        n_valid = 0
        for comp_id in unique_compounds:
            smiles = smiles_dict.get(comp_id)
            if smiles and pd.notna(smiles):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        self.mol_cache[comp_id] = mol
                        n_valid += 1
                except:
                    pass
        
        print(f"  ✓ Cached {n_valid}/{len(unique_compounds)} molecules")
        
        # Create valid sample mask
        self.valid_mask = np.array([
            comp_id in self.mol_cache 
            for comp_id in self.compound_ids
        ])
        
        n_valid_samples = self.valid_mask.sum()
        if n_valid_samples < len(self.valid_mask):
            print(f"  ⚠️  {len(self.valid_mask) - n_valid_samples} samples without valid SMILES (will be skipped)")
    
    def __len__(self):
        return self.valid_mask.sum()
    
    def __getitem__(self, idx):
        """
        Get item by index (only valid samples).
        
        Returns
        -------
        tuple
            (x_transcript, mol_graph, y_label)
        """
        # Map to valid index
        valid_indices = np.where(self.valid_mask)[0]
        actual_idx = valid_indices[idx]
        
        # Get transcriptome data
        x_transcript = torch.FloatTensor(self.X[actual_idx])
        y_label = torch.LongTensor([self.y[actual_idx]])[0]
        
        # Get molecule
        comp_id = self.compound_ids[actual_idx]
        mol = self.mol_cache[comp_id]
        
        # Featurize molecule (will be batched later)
        mol_graph = self.mol_featurizer(mol)
        
        return x_transcript, mol_graph, y_label


def collate_two_tower(batch):
    """
    Custom collate function for batching.
    
    Parameters
    ----------
    batch : list
        List of (x_transcript, mol_graph, y_label) tuples
    
    Returns
    -------
    tuple
        (x_transcript_batch, bmg_batch, y_batch)
    """
    from chemprop.data import BatchMolGraph
    
    # Separate components
    x_transcripts = []
    mol_graphs = []
    y_labels = []
    
    for x_t, mol_g, y in batch:
        x_transcripts.append(x_t)
        mol_graphs.append(mol_g)
        y_labels.append(y)
    
    # Stack transcriptome data
    x_batch = torch.stack(x_transcripts)
    y_batch = torch.stack(y_labels)
    
    # Create batch mol graph
    bmg_batch = BatchMolGraph(mol_graphs)
    
    return x_batch, bmg_batch, y_batch