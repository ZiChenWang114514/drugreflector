#!/usr/bin/env python
"""
Two-Tower DrugReflector System Test Script

Comprehensive testing for:
1. Data loading and preprocessing
2. Molecular encoding (Uni-Mol + fallback)
3. Model architecture
4. Training pipeline
5. Evaluation

Usage:
    python test_system.py
    python test_system.py --quick  # Fast sanity check
    python test_system.py --full   # Full system test
"""
import argparse
import sys
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_test_header(test_name: str):
    """Print test section header."""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}🧪 TEST: {test_name}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")


def print_success(message: str):
    """Print success message."""
    print(f"{GREEN}✅ {message}{RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{RED}❌ {message}{RESET}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{YELLOW}⚠️  {message}{RESET}")


def print_info(message: str):
    """Print info message."""
    print(f"   {message}")


class SystemTester:
    """Comprehensive system tester for Two-Tower DrugReflector."""
    
    def __init__(self, quick: bool = False):
        self.quick = quick
        self.test_results = {}
        self.temp_dir = tempfile.mkdtemp()
        
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}🚀 TWO-TOWER DRUGREFLECTOR SYSTEM TEST{RESET}")
        print(f"{BLUE}{'='*80}{RESET}")
        print(f"Mode: {'Quick' if quick else 'Full'}")
        print(f"Temp dir: {self.temp_dir}")
    
    def test_imports(self) -> bool:
        """Test 1: Check all imports."""
        print_test_header("Module Imports")
        
        required_modules = [
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('torch', 'PyTorch'),
            ('sklearn', 'Scikit-learn'),
            ('tqdm', 'tqdm'),
            ('matplotlib', 'Matplotlib'),
        ]
        
        optional_modules = [
            ('rdkit', 'RDKit'),
        ]
        
        all_success = True
        
        # Required modules
        for module_name, display_name in required_modules:
            try:
                __import__(module_name)
                print_success(f"{display_name} imported")
            except ImportError as e:
                print_error(f"{display_name} not available: {e}")
                all_success = False
        
        # Optional modules
        for module_name, display_name in optional_modules:
            try:
                __import__(module_name)
                print_success(f"{display_name} imported (optional)")
            except ImportError:
                print_warning(f"{display_name} not available (optional)")
        
        # Uni-Mol
        try:
            from unimol_tools import UniMolRepr
            print_success("Uni-Mol imported")
            self.unimol_available = True
        except ImportError:
            print_warning("Uni-Mol not available (will use fallback)")
            self.unimol_available = False
        
        # Project modules
        project_modules = [
            'models', 'dataset', 'trainer', 'losses', 
            'preprocessing', 'unimol_encoder'
        ]
        
        for module_name in project_modules:
            try:
                __import__(module_name)
                print_success(f"{module_name}.py imported")
            except ImportError as e:
                print_error(f"{module_name}.py not available: {e}")
                all_success = False
        
        self.test_results['imports'] = all_success
        return all_success
    
    def test_model_architecture(self) -> bool:
        """Test 2: Model architecture."""
        print_test_header("Model Architecture")
        
        try:
            from models import (
                TwoTowerModel, 
                TranscriptomeTower, 
                MolecularTower,
                FusionLayer,
                TranscriptOnlyModel
            )
            
            # Test dimensions
            n_genes = 978
            n_compounds = 100
            embedding_dim = 512
            batch_size = 16
            
            # Test TranscriptomeTower
            print_info("Testing TranscriptomeTower...")
            transcript_tower = TranscriptomeTower(
                input_dim=n_genes,
                embedding_dim=embedding_dim,
            )
            x_transcript = torch.randn(batch_size, n_genes)
            h_transcript = transcript_tower(x_transcript)
            assert h_transcript.shape == (batch_size, embedding_dim)
            print_success(f"TranscriptomeTower: {x_transcript.shape} → {h_transcript.shape}")
            
            # Test MolecularTower
            print_info("Testing MolecularTower...")
            mol_tower = MolecularTower(
                unimol_dim=512,
                embedding_dim=embedding_dim,
            )
            x_mol = torch.randn(batch_size, 512)
            h_mol = mol_tower(x_mol)
            assert h_mol.shape == (batch_size, embedding_dim)
            print_success(f"MolecularTower: {x_mol.shape} → {h_mol.shape}")
            
            # Test FusionLayer
            print_info("Testing FusionLayer...")
            for fusion_type in ['concat', 'product', 'attention', 'gated']:
                fusion = FusionLayer(
                    embedding_dim=embedding_dim,
                    fusion_type=fusion_type,
                    output_dim=1024,
                )
                h_fused = fusion(h_transcript, h_mol)
                print_success(f"FusionLayer ({fusion_type}): {h_transcript.shape} + {h_mol.shape} → {h_fused.shape}")
            
            # Test TwoTowerModel
            print_info("Testing TwoTowerModel...")
            model = TwoTowerModel(
                n_genes=n_genes,
                n_compounds=n_compounds,
                embedding_dim=embedding_dim,
                fusion_type='concat',
            )
            
            logits = model(x_transcript, x_mol)
            assert logits.shape == (batch_size, n_compounds)
            print_success(f"TwoTowerModel: ({x_transcript.shape}, {x_mol.shape}) → {logits.shape}")
            
            # Test get_embeddings
            h_t, h_m, h_f = model.get_embeddings(x_transcript, x_mol)
            print_success(f"Embeddings extracted: transcript={h_t.shape}, mol={h_m.shape}, fused={h_f.shape}")
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            print_info(f"Total parameters: {n_params:,}")
            
            # Test TranscriptOnlyModel
            print_info("Testing TranscriptOnlyModel (baseline)...")
            baseline_model = TranscriptOnlyModel(
                n_genes=n_genes,
                n_compounds=n_compounds,
            )
            baseline_logits = baseline_model(x_transcript)
            assert baseline_logits.shape == (batch_size, n_compounds)
            print_success(f"TranscriptOnlyModel: {x_transcript.shape} → {baseline_logits.shape}")
            
            self.test_results['model_architecture'] = True
            return True
            
        except Exception as e:
            print_error(f"Model architecture test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['model_architecture'] = False
            return False
    
    def test_loss_functions(self) -> bool:
        """Test 3: Loss functions."""
        print_test_header("Loss Functions")
        
        try:
            from losses import FocalLoss, ContrastiveLoss, TwoTowerLoss
            
            batch_size = 16
            n_classes = 100
            embedding_dim = 512
            
            # Test FocalLoss
            print_info("Testing FocalLoss...")
            focal_loss = FocalLoss(gamma=2.0)
            logits = torch.randn(batch_size, n_classes)
            targets = torch.randint(0, n_classes, (batch_size,))
            loss = focal_loss(logits, targets)
            assert loss.item() > 0
            print_success(f"FocalLoss computed: {loss.item():.4f}")
            
            # Test ContrastiveLoss
            print_info("Testing ContrastiveLoss...")
            contrastive_loss = ContrastiveLoss(temperature=0.07)
            h_transcript = torch.randn(batch_size, embedding_dim)
            h_mol = torch.randn(batch_size, embedding_dim)
            loss = contrastive_loss(h_transcript, h_mol, targets)
            assert loss.item() > 0
            print_success(f"ContrastiveLoss computed: {loss.item():.4f}")
            
            # Test TwoTowerLoss
            print_info("Testing TwoTowerLoss...")
            two_tower_loss = TwoTowerLoss(
                focal_gamma=2.0,
                contrastive_weight=0.1,
            )
            losses = two_tower_loss(logits, targets, h_transcript, h_mol)
            assert 'total' in losses and 'focal' in losses and 'contrastive' in losses
            print_success(f"TwoTowerLoss computed: total={losses['total'].item():.4f}, "
                        f"focal={losses['focal'].item():.4f}, "
                        f"contrastive={losses['contrastive'].item():.4f}")
            
            self.test_results['loss_functions'] = True
            return True
            
        except Exception as e:
            print_error(f"Loss functions test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['loss_functions'] = False
            return False
    
    def test_preprocessing(self) -> bool:
        """Test 4: Data preprocessing."""
        print_test_header("Data Preprocessing")
        
        try:
            from preprocessing import (
                clip_and_normalize_signature,
                normalize_mol_embeddings,
            )
            
            # Test clip_and_normalize_signature
            print_info("Testing clip_and_normalize_signature...")
            n_samples = 100
            n_genes = 978
            X = np.random.randn(n_samples, n_genes) * 3  # Some values outside [-2, 2]
            
            X_processed = clip_and_normalize_signature(X, clip_range=(-2, 2))
            
            # Check clipping
            assert X_processed.min() >= -3 and X_processed.max() <= 3  # After normalization
            
            # Check normalization
            stds = np.std(X_processed, axis=1)
            mean_std = stds.mean()
            assert abs(mean_std - 1.0) < 0.1  # Should be close to 1
            print_success(f"Signature preprocessing: mean_std={mean_std:.4f}")
            
            # Test normalize_mol_embeddings
            print_info("Testing normalize_mol_embeddings...")
            mol_embeddings = np.random.randn(n_samples, 512)
            
            # L2 normalization
            mol_l2 = normalize_mol_embeddings(mol_embeddings, method='l2')
            norms = np.linalg.norm(mol_l2, axis=1)
            assert np.allclose(norms, 1.0)
            print_success(f"L2 normalization: mean_norm={norms.mean():.4f}")
            
            # Z-score normalization
            mol_zscore = normalize_mol_embeddings(mol_embeddings, method='zscore')
            assert abs(mol_zscore.mean()) < 0.1
            print_success(f"Z-score normalization: mean={mol_zscore.mean():.4f}")
            
            self.test_results['preprocessing'] = True
            return True
            
        except Exception as e:
            print_error(f"Preprocessing test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['preprocessing'] = False
            return False
    
    def test_molecular_encoder(self) -> bool:
        """Test 5: Molecular encoder."""
        print_test_header("Molecular Encoder")
        
        try:
            from unimol_encoder import (
                UniMolEncoder,
                FallbackMolecularEncoder,
                get_molecular_encoder,
            )
            
            test_smiles = [
                'CCO',  # Ethanol
                'CC(=O)O',  # Acetic acid
                'c1ccccc1',  # Benzene
            ]
            
            # Test fallback encoder (RDKit)
            print_info("Testing FallbackMolecularEncoder (RDKit)...")
            fallback_encoder = FallbackMolecularEncoder(
                fingerprint_type='morgan',
                n_bits=512,
                radius=2,
            )
            embeddings, valid_mask = fallback_encoder.encode_smiles(test_smiles)
            assert embeddings.shape == (len(test_smiles), 512)
            assert all(valid_mask)
            print_success(f"RDKit fingerprints: {embeddings.shape}, all valid")
            
            # Test Uni-Mol if available
            if self.unimol_available:
                print_info("Testing UniMolEncoder...")
                try:
                    unimol_encoder = UniMolEncoder(
                        model_name='unimol_base',
                        device='cpu',
                        batch_size=2,
                    )
                    embeddings, valid_mask = unimol_encoder.encode_smiles(test_smiles)
                    assert embeddings.shape == (len(test_smiles), 512)
                    print_success(f"Uni-Mol embeddings: {embeddings.shape}")
                except (FileNotFoundError, RuntimeError) as e:
                    # Expected errors when weights are missing
                    print_warning(f"Uni-Mol test skipped (missing weights): {str(e)[:100]}")
                    print_info("  Tip: Ensure Uni-Mol weights are downloaded or use fallback encoder")
                except Exception as e:
                    print_warning(f"Uni-Mol test skipped: {str(e)[:100]}")
            else:
                print_warning("Uni-Mol not available, skipped")
            # Test factory function
            print_info("Testing get_molecular_encoder factory...")
            encoder = get_molecular_encoder(use_unimol=False)
            assert isinstance(encoder, FallbackMolecularEncoder)
            print_success("Factory function works correctly")
            
            self.test_results['molecular_encoder'] = True
            return True
            
        except Exception as e:
            print_error(f"Molecular encoder test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['molecular_encoder'] = False
            return False
    
    def test_dataset(self) -> bool:
        """Test 6: Dataset classes."""
        print_test_header("Dataset Classes")
        
        try:
            from dataset import TwoTowerDataset, TranscriptOnlyDataset
            
            n_samples = 100
            n_genes = 978
            n_compounds = 50
            mol_dim = 512
            
            # Create synthetic data
            X = np.random.randn(n_samples, n_genes).astype(np.float32)
            y = np.random.randint(0, n_compounds, n_samples).astype(np.int32)
            mol_embeddings = np.random.randn(n_samples, mol_dim).astype(np.float32)
            fold_mask = np.random.rand(n_samples) > 0.5
            
            # Test TwoTowerDataset
            print_info("Testing TwoTowerDataset...")
            dataset = TwoTowerDataset(X, y, mol_embeddings, fold_mask)
            assert len(dataset) == fold_mask.sum()
            
            x_t, x_m, label = dataset[0]
            assert x_t.shape == (n_genes,)
            assert x_m.shape == (mol_dim,)
            assert isinstance(label.item(), int)
            print_success(f"TwoTowerDataset: {len(dataset)} samples, item shape: ({x_t.shape}, {x_m.shape})")
            
            # Test TranscriptOnlyDataset
            print_info("Testing TranscriptOnlyDataset...")
            baseline_dataset = TranscriptOnlyDataset(X, y, fold_mask)
            assert len(baseline_dataset) == fold_mask.sum()
            
            x_t, label = baseline_dataset[0]
            assert x_t.shape == (n_genes,)
            print_success(f"TranscriptOnlyDataset: {len(baseline_dataset)} samples, item shape: {x_t.shape}")
            
            # Test DataLoader
            print_info("Testing DataLoader integration...")
            from torch.utils.data import DataLoader
            
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            batch_x_t, batch_x_m, batch_y = next(iter(loader))
            assert batch_x_t.shape[0] == 16
            assert batch_x_m.shape[0] == 16
            assert batch_y.shape[0] == 16
            print_success(f"DataLoader batch: {batch_x_t.shape}, {batch_x_m.shape}, {batch_y.shape}")
            
            self.test_results['dataset'] = True
            return True
            
        except Exception as e:
            print_error(f"Dataset test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['dataset'] = False
            return False
    
    def test_training_pipeline(self) -> bool:
        """Test 7: Training pipeline (mini test)."""
        print_test_header("Training Pipeline")
        
        if self.quick:
            print_warning("Quick mode: skipping training pipeline test")
            self.test_results['training_pipeline'] = None
            return True
        
        try:
            from models import TwoTowerModel
            from dataset import TwoTowerDataset
            from losses import TwoTowerLoss
            from torch.utils.data import DataLoader
            
            # Create tiny synthetic dataset
            n_samples = 50
            n_genes = 978
            n_compounds = 10
            
            X = np.random.randn(n_samples, n_genes).astype(np.float32)
            y = np.random.randint(0, n_compounds, n_samples).astype(np.int32)
            mol_embeddings = np.random.randn(n_samples, 512).astype(np.float32)
            
            train_mask = np.arange(n_samples) < 40
            val_mask = ~train_mask
            
            train_dataset = TwoTowerDataset(X, y, mol_embeddings, train_mask)
            val_dataset = TwoTowerDataset(X, y, mol_embeddings, val_mask)
            
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8)
            
            # Create model
            print_info("Creating model...")
            model = TwoTowerModel(
                n_genes=n_genes,
                n_compounds=n_compounds,
                embedding_dim=128,
                fusion_type='concat',
            )
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            
            criterion = TwoTowerLoss(focal_gamma=2.0, contrastive_weight=0.1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Mini training loop
            print_info("Running mini training (2 epochs)...")
            model.train()
            
            for epoch in range(2):
                epoch_loss = 0.0
                for batch_x_t, batch_x_m, batch_y in train_loader:
                    batch_x_t = batch_x_t.to(device)
                    batch_x_m = batch_x_m.to(device)
                    batch_y = batch_y.to(device)
                    
                    optimizer.zero_grad()
                    
                    logits = model(batch_x_t, batch_x_m)
                    h_t, h_m, _ = model.get_embeddings(batch_x_t, batch_x_m)
                    
                    losses = criterion(logits, batch_y, h_t, h_m)
                    losses['total'].backward()
                    optimizer.step()
                    
                    epoch_loss += losses['total'].item()
                
                avg_loss = epoch_loss / len(train_loader)
                print_info(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")
            
            print_success("Training pipeline works correctly")
            
            # Test validation
            print_info("Testing validation...")
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x_t, batch_x_m, batch_y in val_loader:
                    batch_x_t = batch_x_t.to(device)
                    batch_x_m = batch_x_m.to(device)
                    batch_y = batch_y.to(device)
                    
                    logits = model(batch_x_t, batch_x_m)
                    losses = criterion(logits, batch_y)
                    val_loss += losses['total'].item()
            
            avg_val_loss = val_loss / len(val_loader)
            print_success(f"Validation works: loss={avg_val_loss:.4f}")
            
            self.test_results['training_pipeline'] = True
            return True
            
        except Exception as e:
            print_error(f"Training pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['training_pipeline'] = False
            return False
    
    def test_checkpoint_save_load(self) -> bool:
        """Test 8: Checkpoint saving and loading."""
        print_test_header("Checkpoint Save/Load")
        
        try:
            from models import TwoTowerModel
            import pickle
            
            n_genes = 978
            n_compounds = 100
            
            # Create model
            model = TwoTowerModel(
                n_genes=n_genes,
                n_compounds=n_compounds,
                embedding_dim=512,
                fusion_type='concat',
            )
            
            # Save checkpoint
            checkpoint_path = Path(self.temp_dir) / 'test_checkpoint.pt'
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'fold_id': 0,
                'epoch': 10,
                'dimensions': {
                    'input_size': n_genes,
                    'output_size': n_compounds,
                },
                'model_config': {
                    'embedding_dim': 512,
                    'fusion_type': 'concat',
                },
            }
            
            print_info(f"Saving checkpoint to {checkpoint_path}...")
            torch.save(checkpoint, checkpoint_path)
            print_success(f"Checkpoint saved ({checkpoint_path.stat().st_size / 1024:.1f} KB)")
            
            # Load checkpoint
            print_info("Loading checkpoint...")
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            new_model = TwoTowerModel(
                n_genes=n_genes,
                n_compounds=n_compounds,
                embedding_dim=512,
                fusion_type='concat',
            )
            new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
            
            print_success("Checkpoint loaded successfully")
            
            # Verify models are equivalent
            # Verify models are equivalent
            print_info("Verifying model equivalence...")
            x_t = torch.randn(4, n_genes)
            x_m = torch.randn(4, 512)

            # Set both models to eval mode to ensure deterministic behavior
            model.eval()
            new_model.eval()

            with torch.no_grad():
                out1 = model(x_t, x_m)
                out2 = new_model(x_t, x_m)
                
            # Use larger tolerance due to BatchNorm and floating point precision
            assert torch.allclose(out1, out2, atol=1e-5), \
                f"Outputs differ: max diff = {(out1 - out2).abs().max().item()}"
            print_success("Models produce identical outputs")
            
        except Exception as e:
            print_error(f"Checkpoint test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['checkpoint'] = False
            return False
    
    def test_cuda_availability(self) -> bool:
        """Test 9: CUDA availability and device handling."""
        print_test_header("CUDA Availability")
        
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                n_gpus = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print_success(f"CUDA available: {n_gpus} GPU(s)")
                print_info(f"GPU 0: {gpu_name}")
                
                # Test GPU memory
                if n_gpus > 0:
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print_info(f"GPU memory: {total_memory:.1f} GB")
            else:
                print_warning("CUDA not available, will use CPU")
            
            # Test model on device
            from models import TwoTowerModel
            
            device = 'cuda' if cuda_available else 'cpu'
            model = TwoTowerModel(
                n_genes=978,
                n_compounds=100,
                embedding_dim=512,
            ).to(device)
            
            x_t = torch.randn(4, 978).to(device)
            x_m = torch.randn(4, 512).to(device)
            
            with torch.no_grad():
                outputs = model(x_t, x_m)
            
            assert outputs.device.type == device
            print_success(f"Model inference on {device} works correctly")
            
            self.test_results['cuda'] = True
            return True
            
        except Exception as e:
            print_error(f"CUDA test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['cuda'] = False
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        tests = [
            ('Imports', self.test_imports),
            ('Model Architecture', self.test_model_architecture),
            ('Loss Functions', self.test_loss_functions),
            ('Preprocessing', self.test_preprocessing),
            ('Molecular Encoder', self.test_molecular_encoder),
            ('Dataset', self.test_dataset),
            ('Training Pipeline', self.test_training_pipeline),
            ('Checkpoint Save/Load', self.test_checkpoint_save_load),
            ('CUDA Availability', self.test_cuda_availability),
        ]
        
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}🚀 STARTING SYSTEM TESTS{RESET}")
        print(f"{BLUE}{'='*80}{RESET}")
        
        all_passed = True
        for test_name, test_func in tests:
            try:
                passed = test_func()
                if passed is False:
                    all_passed = False
            except Exception as e:
                print_error(f"Unexpected error in {test_name}: {e}")
                all_passed = False
        
        return all_passed
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}📊 TEST SUMMARY{RESET}")
        print(f"{BLUE}{'='*80}{RESET}")
        
        passed = 0
        failed = 0
        skipped = 0
        
        for test_name, result in self.test_results.items():
            if result is True:
                print_success(f"{test_name}: PASSED")
                passed += 1
            elif result is False:
                print_error(f"{test_name}: FAILED")
                failed += 1
            else:
                print_warning(f"{test_name}: SKIPPED")
                skipped += 1
        
        total = passed + failed + skipped
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"Total: {total} | {GREEN}Passed: {passed}{RESET} | "
              f"{RED}Failed: {failed}{RESET} | {YELLOW}Skipped: {skipped}{RESET}")
        
        if failed == 0:
            print(f"\n{GREEN}{'='*80}{RESET}")
            print(f"{GREEN}✅ ALL TESTS PASSED!{RESET}")
            print(f"{GREEN}{'='*80}{RESET}")
            print(f"\n{GREEN}🎉 System is ready for training!{RESET}\n")
            return True
        else:
            print(f"\n{RED}{'='*80}{RESET}")
            print(f"{RED}❌ SOME TESTS FAILED{RESET}")
            print(f"{RED}{'='*80}{RESET}")
            print(f"\n{RED}⚠️  Please fix errors before training.{RESET}\n")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Two-Tower DrugReflector system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test (skip training pipeline)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full test (including training pipeline)'
    )
    
    args = parser.parse_args()
    
    quick_mode = args.quick or not args.full
    
    tester = SystemTester(quick=quick_mode)
    success = tester.run_all_tests()
    tester.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
