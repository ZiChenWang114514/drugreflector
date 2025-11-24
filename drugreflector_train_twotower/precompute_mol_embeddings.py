#!/usr/bin/env python
"""
Precompute Uni-Mol Molecular Embeddings

This script precomputes molecular embeddings from SMILES strings
using Uni-Mol. Run this once before training.

Usage:
    python precompute_mol_embeddings.py \
        --compound-info /path/to/compoundinfo_beta.txt \
        --output /path/to/mol_embeddings.pkl \
        --batch-size 32

For HPC:
    python precompute_mol_embeddings.py \
        --compound-info ../datasets/LINCS2020/compoundinfo_beta.txt \
        --output ../datasets/mol_embeddings_unimol.pkl \
        --batch-size 64 \
        --device cuda
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Precompute Uni-Mol molecular embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--compound-info',
        type=str,
        required=True,
        help='Path to compoundinfo_beta.txt'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for embeddings pickle'
    )
    
    parser.add_argument(
        '--training-data',
        type=str,
        default=None,
        help='Path to training data (optional, to filter compounds)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device for encoding'
    )
    
    parser.add_argument(
        '--pert-id-col',
        type=str,
        default='pert_id',
        help='Column name for compound IDs'
    )
    
    parser.add_argument(
        '--smiles-col',
        type=str,
        default='canonical_smiles',
        help='Column name for SMILES strings'
    )
    
    parser.add_argument(
        '--use-fallback',
        action='store_true',
        help='Use RDKit fingerprints as fallback if Uni-Mol unavailable'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🧪 PRECOMPUTE MOLECULAR EMBEDDINGS")
    print(f"{'='*80}")
    print(f"  Compound info: {args.compound_info}")
    print(f"  Output: {args.output}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    
    # Load compound info
    print(f"\n📖 Loading compound info...")
    compound_info = pd.read_csv(args.compound_info, sep='\t')
    print(f"  ✓ Loaded {len(compound_info):,} compounds")
    
    # Filter to training compounds if provided
    if args.training_data:
        print(f"\n📖 Loading training data to filter compounds...")
        with open(args.training_data, 'rb') as f:
            training_data = pickle.load(f)
        
        training_compounds = set(training_data['sample_meta']['pert_id'].unique())
        compound_info = compound_info[
            compound_info[args.pert_id_col].isin(training_compounds)
        ]
        print(f"  ✓ Filtered to {len(compound_info):,} training compounds")
    
    # Get unique compounds with valid SMILES
    valid_mask = compound_info[args.smiles_col].notna()
    valid_compounds = compound_info[valid_mask].drop_duplicates(subset=[args.pert_id_col])
    
    print(f"\n📊 Compounds to encode: {len(valid_compounds):,}")
    
    # Try to import Uni-Mol
    try:
        from unimol_encoder import UniMolEncoder
        encoder = UniMolEncoder(
            model_name='unimol_base',
            device=args.device,
            batch_size=args.batch_size,
        )
        use_unimol = True
    except ImportError as e:
        if args.use_fallback:
            print(f"\n⚠️ Uni-Mol not available, using RDKit fingerprints")
            from unimol_encoder import FallbackMolecularEncoder
            encoder = FallbackMolecularEncoder(
                fingerprint_type='morgan',
                n_bits=512,
                radius=2,
            )
            use_unimol = False
        else:
            print(f"\n❌ Uni-Mol not available and --use-fallback not specified")
            print(f"   Error: {e}")
            print(f"\n   Install Uni-Mol with: pip install unimol-tools")
            print(f"   Or use --use-fallback for RDKit fingerprints")
            return
    
    # Encode compounds
    smiles_list = valid_compounds[args.smiles_col].tolist()
    pert_ids = valid_compounds[args.pert_id_col].tolist()
    
    embeddings, valid_mask = encoder.encode_smiles(smiles_list)
    
    # Build dictionary
    pert_to_embedding = {}
    n_valid = 0
    n_invalid = 0
    
    for i, (pert_id, valid) in enumerate(zip(pert_ids, valid_mask)):
        if valid:
            pert_to_embedding[pert_id] = embeddings[i]
            n_valid += 1
        else:
            n_invalid += 1
    
    print(f"\n📊 Encoding Results:")
    print(f"  ✓ Valid embeddings: {n_valid:,}")
    if n_invalid > 0:
        print(f"  ⚠️ Failed: {n_invalid:,}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(pert_to_embedding, f, protocol=4)
    
    file_size_mb = output_path.stat().st_size / (1024**2)
    
    print(f"\n{'='*80}")
    print(f"✅ PRECOMPUTATION COMPLETE!")
    print(f"{'='*80}")
    print(f"  Output: {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Compounds: {len(pert_to_embedding):,}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Encoder: {'Uni-Mol' if use_unimol else 'RDKit fingerprints'}")
    
    # Verify
    sample_pert = list(pert_to_embedding.keys())[0]
    sample_emb = pert_to_embedding[sample_pert]
    print(f"\n📊 Sample embedding:")
    print(f"  Compound: {sample_pert}")
    print(f"  Shape: {sample_emb.shape}")
    print(f"  Range: [{sample_emb.min():.4f}, {sample_emb.max():.4f}]")
    print(f"  Mean: {sample_emb.mean():.4f}")


if __name__ == "__main__":
    main()
