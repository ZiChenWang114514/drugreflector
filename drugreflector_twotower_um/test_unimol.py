#!/usr/bin/env python
"""
Uni-Mol Diagnostic Script

Test Uni-Mol encoder step by step to identify issues.
"""
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔍 UNI-MOL DIAGNOSTIC SCRIPT")
print("="*80)

# Test 1: Import Uni-Mol
print("\n[Test 1] Importing Uni-Mol...")
try:
    from unimol_tools import UniMolRepr
    print("  ✅ Uni-Mol imported successfully")
except ImportError as e:
    print(f"  ❌ Failed to import Uni-Mol: {e}")
    sys.exit(1)

# Test 2: Initialize encoder
print("\n[Test 2] Initializing encoder...")
try:
    encoder = UniMolRepr(data_type='molecule', remove_hs=False)
    print("  ✅ Encoder initialized")
except Exception as e:
    print(f"  ❌ Failed to initialize: {e}")
    sys.exit(1)

# Test 3: Test with simple SMILES
print("\n[Test 3] Testing with single simple SMILES...")
test_smiles = ["CCO"]  # Ethanol
print(f"  SMILES: {test_smiles[0]}")

try:
    print("  Calling encoder.get_repr()...")
    reprs = encoder.get_repr(test_smiles, return_atomic_reprs=False)
    print(f"  ✅ Got representations")
    print(f"  Type: {type(reprs)}")
    
    if isinstance(reprs, dict):
        print(f"  Keys: {list(reprs.keys())}")
        for key, value in reprs.items():
            print(f"    - {key}: type={type(value)}, ", end="")
            if hasattr(value, 'shape'):
                print(f"shape={value.shape}, dtype={value.dtype}")
            elif hasattr(value, '__len__'):
                print(f"len={len(value)}")
            else:
                print(f"value={value}")
    else:
        print(f"  Unexpected type: {type(reprs)}")
        
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Extract CLS representation
print("\n[Test 4] Extracting CLS representation...")
try:
    import numpy as np
    
    if isinstance(reprs, dict):
        print(f"  Format: Dictionary")
        if 'cls_repr' in reprs:
            cls_repr = reprs['cls_repr']
            print(f"  ✅ Found cls_repr")
            print(f"  Type: {type(cls_repr)}")
            print(f"  Shape: {cls_repr.shape if hasattr(cls_repr, 'shape') else 'N/A'}")
            
            # Try to convert to numpy
            if isinstance(cls_repr, np.ndarray):
                print(f"  Already numpy array")
                embedding = cls_repr[0]
            else:
                print(f"  Converting to numpy...")
                embedding = cls_repr.cpu().numpy()[0]
            
            print(f"  ✅ Embedding shape: {embedding.shape}")
            print(f"  Embedding dtype: {embedding.dtype}")
            print(f"  First 5 values: {embedding[:5]}")
        else:
            print(f"  ❌ 'cls_repr' not found in reprs")
            print(f"  Available keys: {list(reprs.keys())}")
            sys.exit(1)
    
    elif isinstance(reprs, (list, tuple)):
        print(f"  Format: List/Tuple (length={len(reprs)})")
        print(f"  ✅ Your Uni-Mol returns list format!")
        
        # First element should be CLS representation
        cls_repr = reprs[0]
        print(f"  Using element 0 as CLS representation")
        print(f"  Type: {type(cls_repr)}")
        print(f"  Shape: {cls_repr.shape if hasattr(cls_repr, 'shape') else 'N/A'}")
        
        # Convert to numpy
        if isinstance(cls_repr, np.ndarray):
            embedding = cls_repr[0]
        else:
            embedding = cls_repr.cpu().numpy()[0]
        
        print(f"  ✅ Embedding shape: {embedding.shape}")
        print(f"  Embedding dtype: {embedding.dtype}")
        print(f"  First 5 values: {embedding[:5]}")
    
    else:
        print(f"  ❌ Unexpected type: {type(reprs)}")
        sys.exit(1)
        
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test with multiple SMILES
print("\n[Test 5] Testing with multiple SMILES (batch)...")
batch_smiles = [
    "CCO",           # Ethanol
    "CC(C)O",        # Isopropanol
    "c1ccccc1",      # Benzene
    "CC(=O)O",       # Acetic acid
]
print(f"  Batch size: {len(batch_smiles)}")

try:
    reprs_batch = encoder.get_repr(batch_smiles, return_atomic_reprs=False)
    
    # Handle different formats
    if isinstance(reprs_batch, dict):
        cls_repr_batch = reprs_batch['cls_repr']
    elif isinstance(reprs_batch, (list, tuple)):
        cls_repr_batch = reprs_batch[0]
    else:
        raise TypeError(f"Unexpected type: {type(reprs_batch)}")
    
    print(f"  ✅ Got batch representations")
    print(f"  Shape: {cls_repr_batch.shape if hasattr(cls_repr_batch, 'shape') else 'N/A'}")
    
    if hasattr(cls_repr_batch, 'shape'):
        expected_shape = (len(batch_smiles), 512)
        if cls_repr_batch.shape == expected_shape:
            print(f"  ✅ Shape is correct: {cls_repr_batch.shape}")
        else:
            print(f"  ⚠️  Shape mismatch: got {cls_repr_batch.shape}, expected {expected_shape}")
            
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test with invalid SMILES
print("\n[Test 6] Testing with invalid SMILES...")
invalid_smiles = ["INVALID_SMILES_XXX"]

try:
    reprs_invalid = encoder.get_repr(invalid_smiles, return_atomic_reprs=False)
    print(f"  ⚠️  Encoder accepted invalid SMILES (may handle internally)")
    if 'cls_repr' in reprs_invalid:
        print(f"  Got cls_repr shape: {reprs_invalid['cls_repr'].shape if hasattr(reprs_invalid['cls_repr'], 'shape') else 'N/A'}")
except Exception as e:
    print(f"  ✅ Encoder rejected invalid SMILES (expected)")
    print(f"  Error: {str(e)[:100]}")

# Test 7: Check GPU usage
print("\n[Test 7] Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ⚠️  CUDA not available (using CPU)")
except Exception as e:
    print(f"  ⚠️  Could not check GPU: {e}")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\n💡 If precompute_mol_embeddings.py still fails:")
print("   1. Check that SMILES data is valid in compoundinfo_beta.txt")
print("   2. Try reducing batch size: --batch-size 8")
print("   3. Check GPU memory: nvidia-smi")
print("   4. Use --use-fallback for RDKit fingerprints instead")
print("\n")