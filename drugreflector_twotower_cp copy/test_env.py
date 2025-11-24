#!/usr/bin/env python
"""
Test environment and dependencies.

Usage:
    python test_env.py
"""
import sys

print("🧪 Testing DrugReflector Two-Tower Environment\n")
print("="*60)

# Test 1: Python version
print("\n1️⃣ Python Version")
print(f"   {sys.version}")
assert sys.version_info >= (3, 7), "❌ Python 3.7+ required"
print("   ✅ OK")

# Test 2: PyTorch
print("\n2️⃣ PyTorch")
try:
    import torch
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
    print("   ✅ OK")
except ImportError as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Core packages
print("\n3️⃣ Core Packages")
required = ['numpy', 'pandas', 'sklearn', 'tqdm', 'matplotlib']
for pkg in required:
    try:
        __import__(pkg)
        print(f"   ✅ {pkg}")
    except ImportError:
        print(f"   ❌ {pkg} - run: pip install {pkg}")

# Test 4: RDKit
print("\n4️⃣ RDKit")
try:
    from rdkit import Chem
    from rdkit import __version__
    print(f"   Version: {__version__}")
    # Test SMILES parsing
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    print("   ✅ OK (SMILES parsing works)")
except ImportError as e:
    print(f"   ❌ FAILED: {e}")
    print("   Install: conda install -c conda-forge rdkit")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 5: Chemprop2
print("\n5️⃣ Chemprop2")
try:
    import chemprop
    from chemprop import nn as cnn
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    print(f"   Version: {chemprop.__version__}")
    
    # Test featurizer
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    mol = Chem.MolFromSmiles("CCO")
    mol_graph = featurizer(mol)
    print(f"   ✅ OK (featurization works)")
except ImportError as e:
    print(f"   ❌ FAILED: {e}")
    print("   Install: pip install chemprop>=2.0.0")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 6: Model import
print("\n6️⃣ Two-Tower Model")
try:
    from models import TwoTowerModel, ChemicalEncoder, TranscriptomeEncoder
    print("   ✅ Model classes imported")
    
    # Test model creation
    model = TwoTowerModel(
        n_genes=978,
        n_compounds=100,
        chem_hidden_dim=512,
        transcript_hidden_dims=[1024, 2048]
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model created ({n_params:,} parameters)")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Summary
print("\n" + "="*60)
print("🎉 Environment test complete!")
print("\nNext steps:")
print("  1. Prepare your data:")
print("     - training_data_lincs2020_final.pkl")
print("     - compoundinfo_beta.txt")
print("  2. Run training:")
print("     python train.py --data-file ... --compound-info ... --fold 0")
print("="*60)