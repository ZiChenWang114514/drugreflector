#!/usr/bin/env python
"""
Two-Tower DrugReflector Training Script

Main script for training Two-Tower DrugReflector models on LINCS 2020 data.
Integrates transcriptome data with molecular embeddings from Uni-Mol.

Usage:
    # Step 1: Precompute molecular embeddings (one-time)
    python precompute_mol_embeddings.py --compound-info compoundinfo_beta.txt --output mol_embeddings.pkl
    
    # Step 2: Train all folds
    python train.py --data-file data.pkl --mol-embeddings mol_embeddings.pkl --output-dir models/ --all-folds
    
    # Or train single fold
    python train.py --data-file data.pkl --mol-embeddings mol_embeddings.pkl --output-dir models/ --fold 0
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from trainer import TwoTowerTrainer
from dataset import load_training_data_with_mol


def print_banner():
    """Print script banner."""
    print(f"\n{'='*80}")
    print(f"🧬 TWO-TOWER DRUGREFLECTOR TRAINING")
    print(f"{'='*80}")
    print(f"  Enhanced with Uni-Mol molecular representations")
    print(f"  Based on Science 2025 paper")


def print_config(args):
    """Print training configuration."""
    print(f"\n📋 Training Configuration:")
    print(f"  Mode: {'Ensemble (all folds)' if args.all_folds else f'Single fold ({args.fold})'}")
    print(f"  Data file: {args.data_file}")
    print(f"  Mol embeddings: {args.mol_embeddings}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Fusion type: {args.fusion_type}")
    print(f"  Initial LR: {args.learning_rate}")
    print(f"  Focal γ: {args.focal_gamma}")
    print(f"  Contrastive weight: {args.contrastive_weight}")
    print(f"  Device: {args.device}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Two-Tower DrugReflector model(s) on LINCS 2020 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ===== Required Arguments =====
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to training data pickle file'
    )
    
    parser.add_argument(
        '--mol-embeddings',
        type=str,
        required=True,
        help='Path to precomputed molecular embeddings pickle file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for models and results'
    )
    
    # ===== Fold Selection =====
    fold_group = parser.add_mutually_exclusive_group()
    fold_group.add_argument(
        '--fold',
        type=int,
        default=None,
        choices=[0, 1, 2],
        help='Train single fold (0, 1, or 2)'
    )
    
    fold_group.add_argument(
        '--all-folds',
        action='store_true',
        help='Train all folds (ensemble training)'
    )
    
    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=None,
        choices=[0, 1, 2],
        help='Train specific folds (e.g., --folds 0 1)'
    )
    
    # ===== Model Architecture =====
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=512,
        help='Embedding dimension for each tower'
    )
    
    parser.add_argument(
        '--fusion-type',
        type=str,
        default='concat',
        choices=['concat', 'product', 'attention', 'gated'],
        help='Fusion strategy for combining tower outputs'
    )
    
    parser.add_argument(
        '--unimol-dim',
        type=int,
        default=512,
        help='Dimension of Uni-Mol embeddings'
    )
    
    # ===== Training Hyperparameters =====
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0139,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--min-lr',
        type=float,
        default=0.00001,
        help='Minimum learning rate for scheduler'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='L2 regularization weight decay'
    )
    
    parser.add_argument(
        '--focal-gamma',
        type=float,
        default=2.0,
        help='Focal loss gamma parameter'
    )
    
    parser.add_argument(
        '--contrastive-weight',
        type=float,
        default=0.1,
        help='Weight for cross-modal contrastive loss (0 to disable)'
    )
    
    parser.add_argument(
        '--t0',
        type=int,
        default=20,
        help='CosineAnnealingWarmRestarts T_0 parameter'
    )
    
    # ===== System Arguments =====
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Training device'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of DataLoader workers'
    )
    
    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--compound-info',
        type=str,
        default=None,
        help='Path to compoundinfo_beta.txt (optional, for computing missing embeddings)'
    )
    
    args = parser.parse_args()
    
    # Validate fold arguments
    if not args.all_folds and args.fold is None and args.folds is None:
        parser.error("Must specify --fold, --folds, or --all-folds")
    
    # Print banner and config
    print_banner()
    print_config(args)
    
    # Load training data with molecular embeddings
    training_data = load_training_data_with_mol(
        training_data_path=args.data_file,
        mol_embeddings_path=args.mol_embeddings,
        compound_info_path=args.compound_info,
        filter_missing_mol=True,
    )
    
    # Create trainer
    trainer = TwoTowerTrainer(
        device=args.device,
        embedding_dim=args.embedding_dim,
        fusion_type=args.fusion_type,
        initial_lr=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        t_0=args.t0,
        focal_gamma=args.focal_gamma,
        contrastive_weight=args.contrastive_weight,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        save_every=args.save_every,
        verbose=not args.quiet,
        unimol_dim=args.unimol_dim,
    )
    
    output_dir = Path(args.output_dir)
    
    # Train based on mode
    if args.all_folds:
        results = trainer.train_all_folds(
            training_data=training_data,
            output_dir=output_dir,
            folds=None,
        )
        
        print(f"\n{'='*80}")
        print(f"✅ ENSEMBLE TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"\n📊 Ensemble Results:")
        for metric, value in results['ensemble_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\n📁 Trained Models:")
        for path in results['model_paths']:
            print(f"  • {path}")
        
    elif args.folds:
        results = trainer.train_all_folds(
            training_data=training_data,
            output_dir=output_dir,
            folds=args.folds,
        )
        
        print(f"\n{'='*80}")
        print(f"✅ MULTI-FOLD TRAINING COMPLETE!")
        print(f"{'='*80}")
        
    else:
        fold_id = args.fold
        result = trainer.train_single_fold(
            training_data=training_data,
            fold_id=fold_id,
            output_dir=output_dir,
        )
        
        print(f"\n{'='*80}")
        print(f"✅ SINGLE FOLD TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"  Fold: {fold_id}")
        print(f"  Best recall: {result['best_recall']:.4f}")
        print(f"  Best epoch: {result['best_epoch'] + 1}")
        print(f"  Model path: {result['model_path']}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
