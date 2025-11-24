"""
Uni-Mol Molecular Encoder for precomputing embeddings.

This module provides utilities for:
1. Loading Uni-Mol model
2. Encoding SMILES strings to embeddings
3. Batch processing for large datasets

Uni-Mol Reference:
- Paper: "Uni-Mol: A Universal 3D Molecular Representation Learning Framework"
- GitHub: https://github.com/dptech-corp/Uni-Mol

Installation:
    pip install unimol-tools
    # or from source:
    # git clone https://github.com/dptech-corp/Uni-Mol.git
    # cd Uni-Mol/unimol_tools && pip install .
"""
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Try importing unimol
UNIMOL_AVAILABLE = False
try:
    from unimol_tools import UniMolRepr
    UNIMOL_AVAILABLE = True
except ImportError:
    pass


class UniMolEncoder:
    """
    Uni-Mol molecular encoder for generating molecular representations.
    
    Uses Uni-Mol to convert SMILES strings to fixed-dimension embeddings.
    
    Parameters
    ----------
    model_name : str
        Uni-Mol model variant: 'unimol_base' (512-dim) or 'unimol_large'
    device : str
        Device for inference ('cuda', 'cpu', or 'auto')
    batch_size : int
        Batch size for encoding
    use_gpu : bool
        Whether to use GPU
    """
    
    def __init__(
        self,
        model_name: str = 'unimol_base',
        device: str = 'auto',
        batch_size: int = 32,
        use_gpu: bool = True,
    ):
        if not UNIMOL_AVAILABLE:
            raise ImportError(
                "Uni-Mol is not installed. Please install via:\n"
                "  pip install unimol-tools\n"
                "Or from source:\n"
                "  git clone https://github.com/dptech-corp/Uni-Mol.git\n"
                "  cd Uni-Mol/unimol_tools && pip install ."
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.use_gpu = use_gpu and (self.device == 'cuda')
        
        print(f"\n{'='*60}")
        print(f"🔬 Initializing Uni-Mol Encoder")
        print(f"{'='*60}")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {batch_size}")
        
        # Initialize Uni-Mol
        # UniMolRepr data_type options: 'molecule', 'oled', 'protein', 'crystal'
        self.encoder = UniMolRepr(data_type='molecule', remove_hs=False)
        
        # Embedding dimension
        self.embedding_dim = 512  # Uni-Mol base output dim
        
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  ✓ Encoder ready")
    
    def encode_smiles(
        self,
        smiles_list: List[str],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, List[bool]]:
        """
        Encode a list of SMILES strings to embeddings.
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings
        show_progress : bool
            Whether to show progress bar
        
        Returns
        -------
        Tuple[np.ndarray, List[bool]]
            - embeddings: Shape (n_molecules, embedding_dim)
            - valid_mask: Boolean list indicating which SMILES were valid
        """
        n_molecules = len(smiles_list)
        embeddings = np.zeros((n_molecules, self.embedding_dim), dtype=np.float32)
        valid_mask = [True] * n_molecules
        
        print(f"\n📊 Encoding {n_molecules:,} molecules...")
        
        # Process in batches
        n_batches = (n_molecules + self.batch_size - 1) // self.batch_size
        
        iterator = range(0, n_molecules, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Encoding")
        
        for start_idx in iterator:
            end_idx = min(start_idx + self.batch_size, n_molecules)
            batch_smiles = smiles_list[start_idx:end_idx]
            
            try:
                # Get representations using Uni-Mol
                # Returns dict with 'cls_repr' (CLS token embedding) and 'atomic_reprs'
                reprs = self.encoder.get_repr(batch_smiles, return_atomic_reprs=False)
                
                # Use CLS representation
                batch_embeddings = reprs['cls_repr']
                
                # Handle numpy array
                if isinstance(batch_embeddings, np.ndarray):
                    embeddings[start_idx:end_idx] = batch_embeddings
                else:
                    embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()
                    
            except Exception as e:
                # Process one by one to identify failures
                for i, smiles in enumerate(batch_smiles):
                    idx = start_idx + i
                    try:
                        repr_single = self.encoder.get_repr([smiles], return_atomic_reprs=False)
                        emb = repr_single['cls_repr']
                        if isinstance(emb, np.ndarray):
                            embeddings[idx] = emb[0]
                        else:
                            embeddings[idx] = emb[0].cpu().numpy()
                    except Exception as e2:
                        valid_mask[idx] = False
                        warnings.warn(f"Failed to encode SMILES at index {idx}: {str(e2)[:50]}")
        
        n_valid = sum(valid_mask)
        n_failed = n_molecules - n_valid
        
        print(f"  ✓ Encoded: {n_valid:,} molecules")
        if n_failed > 0:
            print(f"  ⚠️ Failed: {n_failed:,} molecules")
        
        return embeddings, valid_mask
    
    def encode_compounds(
        self,
        compound_info: pd.DataFrame,
        pert_id_col: str = 'pert_id',
        smiles_col: str = 'canonical_smiles',
        save_path: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Encode all compounds in a compound info DataFrame.
        
        Parameters
        ----------
        compound_info : pd.DataFrame
            DataFrame with compound information
        pert_id_col : str
            Column name for compound IDs
        smiles_col : str
            Column name for SMILES strings
        save_path : Path, optional
            Path to save embeddings
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping pert_id to embedding
        """
        print(f"\n{'='*60}")
        print(f"📊 Encoding Compounds from DataFrame")
        print(f"{'='*60}")
        print(f"  Total compounds: {len(compound_info):,}")
        
        # Filter valid SMILES
        valid_mask = compound_info[smiles_col].notna()
        valid_df = compound_info[valid_mask].copy()
        
        print(f"  Compounds with valid SMILES: {len(valid_df):,}")
        
        # Get unique compounds
        unique_compounds = valid_df.drop_duplicates(subset=[pert_id_col])
        print(f"  Unique compounds: {len(unique_compounds):,}")
        
        # Encode
        smiles_list = unique_compounds[smiles_col].tolist()
        pert_ids = unique_compounds[pert_id_col].tolist()
        
        embeddings, encode_valid = self.encode_smiles(smiles_list)
        
        # Build dictionary
        pert_to_embedding = {}
        for i, (pert_id, valid) in enumerate(zip(pert_ids, encode_valid)):
            if valid:
                pert_to_embedding[pert_id] = embeddings[i]
        
        print(f"\n  ✓ Final embeddings: {len(pert_to_embedding):,} compounds")
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'wb') as f:
                pickle.dump(pert_to_embedding, f, protocol=4)
            
            print(f"  💾 Saved to: {save_path}")
        
        return pert_to_embedding


def precompute_unimol_embeddings(
    compound_info_path: str,
    output_path: str,
    pert_id_col: str = 'pert_id',
    smiles_col: str = 'canonical_smiles',
    batch_size: int = 32,
    device: str = 'auto',
) -> Dict[str, np.ndarray]:
    """
    Convenience function to precompute Uni-Mol embeddings from compound info file.
    
    Parameters
    ----------
    compound_info_path : str
        Path to compoundinfo_beta.txt
    output_path : str
        Path to save embeddings pickle file
    pert_id_col : str
        Column name for compound IDs
    smiles_col : str
        Column name for SMILES strings
    batch_size : int
        Batch size for encoding
    device : str
        Device ('cuda', 'cpu', 'auto')
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping pert_id to embedding
    """
    print(f"\n{'='*80}")
    print(f"🧪 PRECOMPUTE UNI-MOL EMBEDDINGS")
    print(f"{'='*80}")
    
    # Load compound info
    print(f"\n📖 Loading compound info from: {compound_info_path}")
    compound_info = pd.read_csv(compound_info_path, sep='\t')
    print(f"  ✓ Loaded {len(compound_info):,} compounds")
    
    # Initialize encoder
    encoder = UniMolEncoder(
        model_name='unimol_base',
        device=device,
        batch_size=batch_size,
    )
    
    # Encode
    pert_to_embedding = encoder.encode_compounds(
        compound_info=compound_info,
        pert_id_col=pert_id_col,
        smiles_col=smiles_col,
        save_path=output_path,
    )
    
    print(f"\n{'='*80}")
    print(f"✅ PRECOMPUTATION COMPLETE!")
    print(f"{'='*80}")
    print(f"  Embeddings saved to: {output_path}")
    print(f"  Total compounds: {len(pert_to_embedding):,}")
    
    return pert_to_embedding


class FallbackMolecularEncoder:
    """
    Fallback molecular encoder when Uni-Mol is not available.
    
    Uses RDKit molecular fingerprints as features.
    This is simpler but less powerful than Uni-Mol.
    """
    
    def __init__(self, fingerprint_type: str = 'morgan', n_bits: int = 512, radius: int = 2):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            self.Chem = Chem
            self.AllChem = AllChem
        except ImportError:
            raise ImportError("RDKit is required for fallback encoder")
        
        self.fingerprint_type = fingerprint_type
        self.n_bits = n_bits
        self.radius = radius
        self.embedding_dim = n_bits
        
        print(f"\n⚠️ Using Fallback Molecular Encoder (RDKit fingerprints)")
        print(f"  Fingerprint: {fingerprint_type}")
        print(f"  Bits: {n_bits}")
    
    def encode_smiles(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[bool]]:
        """Encode SMILES using RDKit fingerprints."""
        n_molecules = len(smiles_list)
        embeddings = np.zeros((n_molecules, self.n_bits), dtype=np.float32)
        valid_mask = [True] * n_molecules
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Computing fingerprints")):
            try:
                mol = self.Chem.MolFromSmiles(smiles)
                if mol is None:
                    valid_mask[i] = False
                    continue
                
                if self.fingerprint_type == 'morgan':
                    fp = self.AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.radius, nBits=self.n_bits
                    )
                else:
                    fp = self.Chem.RDKFingerprint(mol, fpSize=self.n_bits)
                
                embeddings[i] = np.array(fp)
                
            except Exception as e:
                valid_mask[i] = False
        
        return embeddings, valid_mask


def get_molecular_encoder(use_unimol: bool = True, **kwargs):
    """
    Factory function to get appropriate molecular encoder.
    
    Parameters
    ----------
    use_unimol : bool
        Whether to use Uni-Mol (if available)
    **kwargs
        Arguments passed to encoder
    
    Returns
    -------
    Encoder
        UniMolEncoder or FallbackMolecularEncoder
    """
    if use_unimol and UNIMOL_AVAILABLE:
        return UniMolEncoder(**kwargs)
    else:
        if use_unimol and not UNIMOL_AVAILABLE:
            print("⚠️ Uni-Mol not available, falling back to RDKit fingerprints")
        return FallbackMolecularEncoder(**kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute Uni-Mol embeddings")
    parser.add_argument('--compound-info', type=str, required=True,
                       help='Path to compoundinfo_beta.txt')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for embeddings pickle')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    precompute_unimol_embeddings(
        compound_info_path=args.compound_info,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
