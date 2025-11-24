#!/usr/bin/env python
"""
DrugReflector Two-Tower Training Script

Integrates transcriptome and chemical structure information.
Uses Chemprop2 MPNN encoder for molecular structures.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from trainer import TwoTowerTrainer


def load_training_data(data_file: Path, compound_info_file: Path) -> dict:
    """Load training data and compound information."""
    print(f"\n{'='*80}")
    print(f"📂 Loading Training Data")
    print(f"{'='*80}")
    
    # Load preprocessed training data
    print(f"  Loading from: {data_file}")
    with open(data_file, 'rb') as f:
        training_data = pickle.load(f)
    
    print(f"  ✓ Data loaded successfully")
    print(f"  Samples: {len(training_data['X']):,}")
    print(f"  Features: {training_data['X'].shape[1]:,}")
    print(f"  Compounds: {len(training_data['compound_names']):,}")
    
    # Load compound info for SMILES
    print(f"\n  Loading compound info from: {compound_info_file}")
    import pandas as pd
    compound_info = pd.read_csv(compound_info_file, sep='\t')
    
    # Create pert_id -> SMILES mapping
    smiles_col = 'canonical_smiles'
    if smiles_col not in compound_info.columns:
        raise ValueError(f"Column '{smiles_col}' not found in compound_info")
    
    smiles_dict = dict(zip(
        compound_info['pert_id'],
        compound_info[smiles_col]
    ))
    
    # Add SMILES to training data
    training_data['smiles_dict'] = smiles_dict
    
    # Validate coverage
    compounds_with_smiles = sum(
        1 for c in training_data['compound_names'] 
        if c in smiles_dict and pd.notna(smiles_dict[c])
    )
    coverage = compounds_with_smiles / len(training_data['compound_names']) * 100
    print(f"  ✓ Compounds with SMILES: {compounds_with_smiles:,}/{len(training_data['compound_names']):,} ({coverage:.1f}%)")
    
    return training_data


def main():
    parser = argparse.ArgumentParser(
        description="Train Two-Tower DrugReflector with Chemical Structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to training data pickle file')
    parser.add_argument('--compound-info', type=str, required=True,
                       help='Path to compoundinfo_beta.txt')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    
    # Fold selection
    fold_group = parser.add_mutually_exclusive_group()
    fold_group.add_argument('--fold', type=int, choices=[0,1,2],
                           help='Train single fold')
    fold_group.add_argument('--all-folds', action='store_true',
                           help='Train all folds')
    parser.add_argument('--folds', type=int, nargs='+', choices=[0,1,2],
                       help='Train specific folds')
    
    # Architecture
    parser.add_argument('--chem-hidden-dim', type=int, default=512,
                       help='Chemical encoder output dimension')
    parser.add_argument('--transcript-hidden-dims', type=int, nargs='+',
                       default=[1024, 2048], help='Transcriptome encoder hidden dims')
    parser.add_argument('--fusion-method', type=str, default='concat',
                       choices=['concat', 'multiply', 'add', 'gated'],
                       help='Feature fusion method')
    parser.add_argument('--mpnn-depth', type=int, default=3,
                       help='MPNN depth')
    parser.add_argument('--mpnn-dropout', type=float, default=0.0,
                       help='MPNN dropout')
    parser.add_argument('--use-3d', action='store_true',
                   help='Use 3D molecular coordinates')
    parser.add_argument('--d-coord', type=int, default=16,
                    help='Dimension of 3D coordinate features (only used if --use-3d)')
    parser.add_argument('--conformer-method', type=str, default='ETKDG',
                    choices=['ETKDG', 'MMFF', 'UFF'],
                    help='3D conformer generation method (only used if --use-3d)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.0139)
    parser.add_argument('--min-lr', type=float, default=0.00001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--t0', type=int, default=20)
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=10)
    
    args = parser.parse_args()
    
    # Validate
    if not args.all_folds and args.fold is None and args.folds is None:
        parser.error("Must specify --fold, --folds, or --all-folds")
    
    # Banner
    print(f"\n{'='*80}")
    print(f"🧬 DRUGREFLECTOR TWO-TOWER TRAINING")
    print(f"{'='*80}")
    print(f"  Transcriptome + Chemical Structure Integration")
    print(f"  Using Chemprop2 MPNN Encoder")
    
    # Load data
    data_file = Path(args.data_file)
    compound_info = Path(args.compound_info)
    training_data = load_training_data(data_file, compound_info)
    
    # Create trainer
    trainer = TwoTowerTrainer(
        device=args.device,
        chem_hidden_dim=args.chem_hidden_dim,
        transcript_hidden_dims=args.transcript_hidden_dims,
        fusion_method=args.fusion_method,
        mpnn_depth=args.mpnn_depth,
        mpnn_dropout=args.mpnn_dropout,
        use_3d=args.use_3d,
        d_coord=args.d_coord,
        conformer_method=args.conformer_method,
        initial_lr=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        t_0=args.t0,
        focal_gamma=args.focal_gamma,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        save_every=args.save_every
    )
    
    output_dir = Path(args.output_dir)
    
    # Train
    if args.all_folds:
        results = trainer.train_all_folds(training_data, output_dir, folds=None)
        print(f"\n✅ ENSEMBLE TRAINING COMPLETE!")
        print(f"📊 Ensemble Metrics:")
        for metric, value in results['ensemble_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    elif args.folds:
        results = trainer.train_all_folds(training_data, output_dir, folds=args.folds)
        print(f"\n✅ MULTI-FOLD TRAINING COMPLETE!")
    else:
        result = trainer.train_single_fold(training_data, args.fold, output_dir)
        print(f"\n✅ SINGLE FOLD TRAINING COMPLETE!")
        print(f"  Best recall: {result['best_recall']:.4f}")
        print(f"  Model: {result['model_path']}")


if __name__ == "__main__":
    main()