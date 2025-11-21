"""
DrugReflector Training Script

Main script for training DrugReflector models on LINCS 2020 data.
Supports both single-fold and multi-fold (ensemble) training.

Usage:
    # Train single fold
    python train.py --data-file data.pkl --output-dir models/fold0 --fold 0
    
    # Train all folds (ensemble)
    python train.py --data-file data.pkl --output-dir models/ensemble --all-folds
    
    # Train specific folds
    python train.py --data-file data.pkl --output-dir models/folds_01 --folds 0 1
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from trainer import DrugReflectorTrainer 

# Add parent directory to path for imports
try:
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    import os
    sys.path.append(str(Path(os.getcwd()).parent))



def load_training_data(data_file: Path) -> dict:
    """
    Load and validate training data.
    
    Parameters
    ----------
    data_file : Path
        Path to training data pickle file
    
    Returns
    -------
    dict
        Training data dictionary
    """
    print(f"\n{'='*80}")
    print(f"📂 Loading Training Data")
    print(f"{'='*80}")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    
    print(f"  Loading from: {data_file}")
    with open(data_file, 'rb') as f:
        training_data = pickle.load(f)
    
    print(f"  ✓ Data loaded successfully")
    print(f"  Keys: {list(training_data.keys())}")
    
    # Validate required keys
    required_keys = ['X', 'y', 'folds', 'compound_names']
    missing_keys = [k for k in required_keys if k not in training_data]
    if missing_keys:
        print(f"❌ Missing required keys: {missing_keys}")
        sys.exit(1)
    
    # Print data summary
    print(f"\n📊 Data Summary:")
    print(f"  Samples: {len(training_data['X']):,}")
    print(f"  Features: {training_data['X'].shape[1]:,}")
    print(f"  Compounds: {len(training_data['compound_names']):,}")
    print(f"  Unique folds: {np.unique(training_data['folds'])}")
    
    return training_data


def print_banner():
    """Print script banner."""
    print(f"\n{'='*80}")
    print(f"🧬 DRUGREFLECTOR TRAINING")
    print(f"{'='*80}")
    print(f"  Based on Science 2025 paper")
    print(f"  Training on LINCS 2020 dataset")


def print_config(args):
    """Print training configuration."""
    print(f"\n📋 Training Configuration:")
    print(f"  Mode: {'Ensemble (all folds)' if args.all_folds else f'Single fold ({args.fold})'}")
    if args.folds:
        print(f"  Folds: {args.folds}")
    print(f"  Data file: {args.data_file}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Initial LR: {args.learning_rate}")
    print(f"  Focal γ: {args.focal_gamma}")
    print(f"  Device: {args.device}")
    print(f"  Workers: {args.num_workers}")


def main():
    parser = argparse.ArgumentParser(
        description="Train DrugReflector model(s) on LINCS 2020 data",
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
    
    # ===== Training Hyperparameters (from SI Table S5) =====
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
        help='Initial learning rate (from SI Table S5)'
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
    
    args = parser.parse_args()
    
    # Validate fold arguments
    if not args.all_folds and args.fold is None and args.folds is None:
        parser.error("Must specify --fold, --folds, or --all-folds")
    
    # Print banner and config
    print_banner()
    print_config(args)
    
    # Load training data
    data_file = Path(args.data_file)
    training_data = load_training_data(data_file)
    
    # Create trainer
    trainer = DrugReflectorTrainer(
        device=args.device,
        initial_lr=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        t_0=args.t0,
        focal_gamma=args.focal_gamma,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        save_every=args.save_every,
        verbose=not args.quiet
    )
    
    output_dir = Path(args.output_dir)
    
    # Train based on mode
    if args.all_folds:
        # Train all folds
        results = trainer.train_all_folds(
            training_data=training_data,
            output_dir=output_dir,
            folds=None  # Train all folds (0, 1, 2)
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
        
        print(f"\n🎯 Next steps:")
        print(f"  1. Review ensemble comparison: {output_dir}/ensemble_comparison.png")
        print(f"  2. Load ensemble for inference:")
        print(f"     from drugreflector import DrugReflector")
        print(f"     model = DrugReflector(checkpoint_paths={results['model_paths']})")
        
    elif args.folds:
        # Train specific folds
        results = trainer.train_all_folds(
            training_data=training_data,
            output_dir=output_dir,
            folds=args.folds
        )
        
        print(f"\n{'='*80}")
        print(f"✅ MULTI-FOLD TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"\n📊 Results:")
        for fold_id, fold_result in results['fold_results'].items():
            print(f"  Fold {fold_id}: Best recall = {fold_result['best_recall']:.4f}")
        
        print(f"\n📁 Trained Models:")
        for path in results['model_paths']:
            print(f"  • {path}")
    
    else:
        # Train single fold
        fold_id = args.fold
        result = trainer.train_single_fold(
            training_data=training_data,
            fold_id=fold_id,
            output_dir=output_dir
        )
        
        print(f"\n{'='*80}")
        print(f"✅ SINGLE FOLD TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"  Fold: {fold_id}")
        print(f"  Best recall: {result['best_recall']:.4f}")
        print(f"  Best epoch: {result['best_epoch'] + 1}/{args.epochs}")
        print(f"  Model path: {result['model_path']}")
        
        print(f"\n🎯 Next steps:")
        print(f"  1. Review training curves: {output_dir}/training_curves_fold_{fold_id}.png")
        print(f"  2. Load model for inference:")
        print(f"     from drugreflector import DrugReflector")
        print(f"     model = DrugReflector(checkpoint_paths=['{result['model_path']}'])")
        print(f"  3. Train other folds for ensemble:")
        print(f"     python train.py --data-file {args.data_file} --output-dir {args.output_dir} --all-folds")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()