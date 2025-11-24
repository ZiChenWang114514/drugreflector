#!/usr/bin/env python
"""
Download Uni-Mol Model Weights

This script downloads Uni-Mol weights to a local directory for offline use.
Run this once on a machine with internet access, then transfer the weights
to your HPC environment.

Usage:
    # Download to default directory
    python download_unimol_weights.py
    
    # Download to custom directory
    python download_unimol_weights.py --output-dir /path/to/weights
    
    # On HPC, specify the weights directory
    python precompute_mol_embeddings.py --weights-dir /path/to/weights ...
"""
import argparse
import os
import shutil
from pathlib import Path
import urllib.request
import warnings
warnings.filterwarnings('ignore')


UNIMOL_WEIGHTS = {
    'mol_pre_all_h_220816.pt': {
        'url': 'https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt',
        'size': '~180MB',
        'description': 'Uni-Mol pretrained model weights',
    },
    'mol.dict.txt': {
        'url': 'https://huggingface.co/dptech/Uni-Mol-Models/blob/main/mol.dict.txt',
        'size': '~1KB',
        'description': 'Uni-Mol vocabulary dictionary',
    },
}


def download_file(url: str, output_path: Path, description: str = ''):
    """Download file with progress bar."""
    print(f"\n📥 Downloading: {output_path.name}")
    if description:
        print(f"   {description}")
    print(f"   URL: {url}")
    
    try:
        # Create a simple progress callback
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 / total_size)
                downloaded_mb = (block_num * block_size) / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"   Progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='\r')
        
        urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
        print()  # New line after progress
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Downloaded: {file_size:.1f} MB")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def verify_weights(weights_dir: Path) -> bool:
    """Verify all required weights exist."""
    print(f"\n🔍 Verifying weights in: {weights_dir}")
    
    all_present = True
    for filename in UNIMOL_WEIGHTS.keys():
        filepath = weights_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {filename} (missing)")
            all_present = False
    
    return all_present


def main():
    parser = argparse.ArgumentParser(
        description="Download Uni-Mol model weights for offline use",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./unimol_weights',
        help='Directory to save weights'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing weights without downloading'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🧪 UNI-MOL WEIGHTS DOWNLOADER")
    print("="*80)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Verify only mode
    if args.verify_only:
        success = verify_weights(output_dir)
        if success:
            print("\n✅ All weights present!")
        else:
            print("\n❌ Some weights missing. Run without --verify-only to download.")
        return
    
    # Download weights
    print(f"\n📦 Downloading {len(UNIMOL_WEIGHTS)} files...")
    
    download_success = {}
    
    for filename, info in UNIMOL_WEIGHTS.items():
        output_path = output_dir / filename
        
        # Skip if exists and not forcing
        if output_path.exists() and not args.force:
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"\n✅ {filename} already exists ({file_size:.1f} MB)")
            print(f"   Use --force to re-download")
            download_success[filename] = True
            continue
        
        # Download
        success = download_file(
            url=info['url'],
            output_path=output_path,
            description=info['description']
        )
        download_success[filename] = success
    
    # Summary
    print("\n" + "="*80)
    print("📊 DOWNLOAD SUMMARY")
    print("="*80)
    
    n_success = sum(download_success.values())
    n_total = len(download_success)
    
    for filename, success in download_success.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {filename}: {status}")
    
    print(f"\nTotal: {n_success}/{n_total} files")
    
    if n_success == n_total:
        print("\n✅ ALL WEIGHTS DOWNLOADED!")
        print(f"\n📁 Weights location: {output_dir.absolute()}")
        print("\n🚀 Next steps:")
        print(f"   1. Transfer to HPC: scp -r {output_dir} user@hpc:/path/to/weights")
        print(f"   2. Use in scripts: --weights-dir /path/to/weights")
        print("\n   Example:")
        print(f"   python precompute_mol_embeddings.py \\")
        print(f"       --compound-info compoundinfo_beta.txt \\")
        print(f"       --output embeddings.pkl \\")
        print(f"       --weights-dir {output_dir.absolute()}")
    else:
        print("\n❌ SOME DOWNLOADS FAILED")
        print("   Check your internet connection and try again")
        print("   You can also download manually from:")
        for filename, info in UNIMOL_WEIGHTS.items():
            if not download_success.get(filename, False):
                print(f"   - {info['url']}")


if __name__ == "__main__":
    main()