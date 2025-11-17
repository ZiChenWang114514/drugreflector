"""
Example script for using trained DrugReflector models for inference.

This demonstrates how to load trained models and make predictions on new data.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from drugreflector import DrugReflector, compute_vscores_adata
from utils import load_h5ad_file


def main():
    parser = argparse.ArgumentParser(
        description="DrugReflector inference example"
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Directory containing trained model checkpoints'
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Input H5AD file with v-scores or gene expression data'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of top predictions to return'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='predictions.csv',
        help='Output file for predictions'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧬 DrugReflector Inference Example")
    print("=" * 80)
    
    # Locate model checkpoints
    model_dir = Path(args.model_dir)
    model_paths = [
        model_dir / 'model_fold_0.pt',
        model_dir / 'model_fold_1.pt',
        model_dir / 'model_fold_2.pt'
    ]
    
    # Check if models exist
    for model_path in model_paths:
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)
    
    # Load DrugReflector
    print(f"\n📂 Loading DrugReflector models...")
    model = DrugReflector(checkpoint_paths=[str(p) for p in model_paths])
    print(f"✓ Models loaded successfully")
    print(f"  Number of compounds: {model.n_compounds:,}")
    
    # Load or create example data
    if args.input_file:
        print(f"\n📂 Loading input data from {args.input_file}...")
        adata = load_h5ad_file(args.input_file)
        print(f"✓ Loaded {adata.n_obs} observations with {adata.n_vars} genes")
        
        # Assume input is v-scores
        vscores = pd.Series(
            adata.X.flatten() if adata.n_obs == 1 else adata.X[0],
            index=adata.var_names,
            name='query'
        )
    else:
        print(f"\n⚠️  No input file provided, creating synthetic example...")
        # Create example v-score vector
        np.random.seed(42)
        n_genes = 978
        vscores = pd.Series(
            np.random.randn(n_genes) * 0.5,
            index=[f'GENE{i}' for i in range(n_genes)],
            name='example_query'
        )
        print(f"✓ Created example v-scores for {len(vscores)} genes")
        print(f"\n💡 In real use, provide v-scores computed from scRNA-seq data:")
        print(f"   vscores = compute_vscores_adata(")
        print(f"       adata,")
        print(f"       group_col='cell_type',")
        print(f"       group1_value='control',")
        print(f"       group2_value='treatment'")
        print(f"   )")
    
    # Make predictions
    print(f"\n🔍 Making predictions (top {args.top_k})...")
    predictions = model.predict(vscores, n_top=args.top_k)
    
    print(f"✓ Predictions complete")
    print(f"  Predicted compounds: {len(predictions)}")
    
    # Display top 10
    print(f"\n{'='*80}")
    print(f"🎯 TOP 10 PREDICTIONS")
    print(f"{'='*80}")
    
    vscore_name = vscores.name
    rank_col = ('rank', vscore_name)
    prob_col = ('prob', vscore_name)
    logit_col = ('logit', vscore_name)
    
    top_10 = predictions.nsmallest(10, rank_col)
    
    print(f"\n{'Rank':<6} {'Compound':<30} {'Probability':<12} {'Logit':<10}")
    print("-" * 80)
    for compound in top_10.index:
        rank = int(top_10.loc[compound, rank_col])
        prob = top_10.loc[compound, prob_col]
        logit = top_10.loc[compound, logit_col]
        print(f"{rank:<6} {compound:<30} {prob:<12.6f} {logit:<10.4f}")
    
    # Save results
    output_path = Path(args.output_file)
    predictions.to_csv(output_path)
    print(f"\n💾 Full predictions saved to: {output_path}")
    
    # Statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"   Mean probability: {predictions[prob_col].mean():.6f}")
    print(f"   Max probability: {predictions[prob_col].max():.6f}")
    print(f"   Compounds with prob > 0.001: {(predictions[prob_col] > 0.001).sum()}")
    
    print(f"\n{'='*80}")
    print(f"✅ Inference Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()