"""
Prepare training data for DrugReflector from LINCS Level 4 data.

This script processes raw LINCS data into the format required for training.
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from drugreflector_training.preprocessing import create_fold_splits


def prepare_training_data(
    expression_file: str,
    metadata_file: str,
    output_file: str,
    n_folds: int = 3,
    random_state: int = 42
):
    """
    Prepare training data from raw LINCS files.
    
    Parameters
    ----------
    expression_file : str
        Path to expression matrix file (.npy or .csv)
    metadata_file : str
        Path to metadata file (.csv)
    output_file : str
        Path to save processed data (.pkl)
    n_folds : int, default=3
        Number of cross-validation folds
    random_state : int, default=42
        Random seed
    """
    print("=" * 80)
    print("📊 Preparing DrugReflector Training Data")
    print("=" * 80)
    
    # Load expression data
    print(f"\n📂 Loading expression data from {expression_file}...")
    expression_path = Path(expression_file)
    
    if expression_path.suffix == '.npy':
        X = np.load(expression_file)
    elif expression_path.suffix == '.csv':
        X = pd.read_csv(expression_file, index_col=0).values
    else:
        raise ValueError("Expression file must be .npy or .csv")
    
    print(f"✓ Loaded expression matrix: {X.shape}")
    
    # Load metadata
    print(f"\n📂 Loading metadata from {metadata_file}...")
    metadata = pd.read_csv(metadata_file)
    print(f"✓ Loaded metadata: {len(metadata)} samples")
    
    # Validate
    if len(metadata) != X.shape[0]:
        raise ValueError(f"Metadata length ({len(metadata)}) doesn't match expression shape ({X.shape[0]})")
    
    # Extract compound labels
    if 'pert_id' not in metadata.columns:
        raise ValueError("Metadata must have 'pert_id' column")
    
    print(f"\n🔍 Processing compound labels...")
    unique_compounds = metadata['pert_id'].unique()
    compound_to_idx = {cpd: i for i, cpd in enumerate(unique_compounds)}
    y = metadata['pert_id'].map(compound_to_idx).values
    
    print(f"✓ Found {len(unique_compounds)} unique compounds")
    print(f"  Samples per compound: {len(y) / len(unique_compounds):.1f} (average)")
    
    # Create fold splits (balanced by compound)
    print(f"\n📋 Creating {n_folds}-fold splits...")
    folds = create_fold_splits(len(y), n_folds=n_folds, random_state=random_state)
    
    # Get gene names if available
    if expression_path.suffix == '.csv':
        gene_names = pd.read_csv(expression_file, index_col=0).columns.tolist()
    else:
        gene_names = [f'gene_{i}' for i in range(X.shape[1])]
    
    # Package training data
    training_data = {
        'X': X,
        'y': y,
        'folds': folds,
        'compound_names': unique_compounds.tolist(),
        'gene_names': gene_names,
        'metadata': metadata
    }
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    print(f"\n💾 Saving training data to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"✓ Training data saved successfully")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Data Preparation Complete!")
    print(f"{'='*80}")
    print(f"\nData Summary:")
    print(f"  Samples: {len(X):,}")
    print(f"  Genes: {X.shape[1]}")
    print(f"  Compounds: {len(unique_compounds):,}")
    print(f"  Folds: {n_folds}")
    print(f"\nOutput file: {output_path}")
    print(f"\nNext step:")
    print(f"  python scripts/train.py --data-file {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DrugReflector training data from LINCS files"
    )
    
    parser.add_argument(
        '--expression-file',
        type=str,
        required=True,
        help='Path to expression matrix (.npy or .csv)'
    )
    
    parser.add_argument(
        '--metadata-file',
        type=str,
        required=True,
        help='Path to metadata file (.csv with pert_id column)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='./data/training_data.pkl',
        help='Path to save processed data (.pkl)'
    )
    
    parser.add_argument(
        '--n-folds',
        type=int,
        default=3,
        help='Number of cross-validation folds (default: 3)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for fold splitting (default: 42)'
    )
    
    args = parser.parse_args()
    
    prepare_training_data(
        expression_file=args.expression_file,
        metadata_file=args.metadata_file,
        output_file=args.output_file,
        n_folds=args.n_folds,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()