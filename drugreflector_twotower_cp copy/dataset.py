"""
PyTorch Dataset for Two-Tower training.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional

try:
    from chemprop.data import MoleculeDatapoint, MoleculeDataset
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False


class TwoTowerDataset(Dataset):
    """
    Dataset combining transcriptome and chemical structure.
    
    Parameters
    ----------
    X : np.ndarray
        Gene expression matrix (n_samples, n_genes)
    y : np.ndarray
        Compound labels (n_samples,)
    compound_ids : np.ndarray
        Compound IDs for each sample (n_samples,)
    smiles_dict : Dict[str, str]
        Mapping from compound ID to SMILES
    fold_mask : np.ndarray, optional
        Boolean mask for samples to include
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        compound_ids: np.ndarray,
        smiles_dict: Dict[str, str],
        fold_mask: Optional[np.ndarray] = None,
        preload_to_gpu: bool = False,
        use_3d: bool = False,
        conformer_method: str = 'ETKDG',
    ):
        if not CHEMPROP_AVAILABLE:
            raise ImportError("Chemprop is required")
        
        self.use_3d = use_3d
        self.conformer_method = conformer_method
        # Apply fold mask
        if fold_mask is not None:
            self.X = X[fold_mask]
            self.y = y[fold_mask]
            self.compound_ids = compound_ids[fold_mask]
        else:
            self.X = X
            self.y = y
            self.compound_ids = compound_ids
        
        self.smiles_dict = smiles_dict
        
        # Create molecule featurizer
        self.mol_featurizer = SimpleMoleculeMolGraphFeaturizer()
        
        print(f"  🔧 Featurizer dimensions:")
        print(f"    Atom feature dim: {self.mol_featurizer.atom_fdim}")
        print(f"    Bond feature dim: {self.mol_featurizer.bond_fdim}")
        
        print(" Pre-featurizing and validating molecules...")
        self.mol_cache = {}
        self.mol_graph_cache = {}  # 缓存MolGraph
        unique_compounds = np.unique(self.compound_ids)
        
        n_valid = 0
        n_invalid_smiles = 0
        n_invalid_mol = 0
        n_invalid_graph = 0
        
        for comp_id in unique_compounds:
            smiles = smiles_dict.get(comp_id)
            
            # 检查点1: SMILES是否存在且有效
            if not smiles or pd.isna(smiles):
                n_invalid_smiles += 1
                continue
            
            try:
                # 检查点2: 能否转换为Mol对象
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    n_invalid_mol += 1
                    print(f"  !  Invalid SMILES for {comp_id}: {smiles}")
                    continue
                
                # 检查点3: 能否特征化
                mol_graph = self.mol_featurizer(mol)
                
                # 检查点4: 特征化结果是否有效
                if mol_graph is None:
                    n_invalid_graph += 1
                    print(f"  !  Featurization returned None for {comp_id}")
                    continue
                
                if not hasattr(mol_graph, 'V') or mol_graph.V is None:
                    n_invalid_graph += 1
                    print(f"  !  MolGraph.V is None for {comp_id}")
                    continue
                
                if not hasattr(mol_graph, 'E') or mol_graph.E is None:
                    n_invalid_graph += 1
                    print(f"  !  MolGraph.E is None for {comp_id}")
                    continue
                
                # 验证特征维度
                expected_atom_dim = self.mol_featurizer.atom_fdim
                expected_bond_dim = self.mol_featurizer.bond_fdim
                
                if mol_graph.V.shape[1] != expected_atom_dim:
                    n_invalid_graph += 1
                    print(f"  !  Wrong atom feature dim for {comp_id}: "
                          f"got {mol_graph.V.shape[1]}, expected {expected_atom_dim}")
                    continue
                
                if mol_graph.E.shape[1] != expected_bond_dim:
                    n_invalid_graph += 1
                    print(f"  !  Wrong bond feature dim for {comp_id}: "
                          f"got {mol_graph.E.shape[1]}, expected {expected_bond_dim}")
                    continue
                
                # 全部通过，缓存
                self.mol_cache[comp_id] = mol
                self.mol_graph_cache[comp_id] = mol_graph
                
                # 如果使用3D，生成坐标
                if self.use_3d:
                    coords_3d = self._generate_conformer(mol)
                    if coords_3d is not None:
                        # 缓存3D坐标（作为torch tensor）
                        if not hasattr(self, 'coords_3d_cache'):
                            self.coords_3d_cache = {}
                        self.coords_3d_cache[comp_id] = torch.FloatTensor(coords_3d)
                    else:
                        print(f"  !  Failed to generate 3D coords for {comp_id}, will skip in training")

                n_valid += 1
                
            except Exception as e:
                n_invalid_graph += 1
                print(f"  !  Error processing {comp_id}: {e}")
                continue
        
        # 报告统计
        print(f"\n Molecule Validation Summary:")
        print(f"    Total unique compounds: {len(unique_compounds)}")
        print(f"    ✓ Valid: {n_valid}")
        print(f"    ✗ Invalid SMILES: {n_invalid_smiles}")
        print(f"    ✗ Invalid Mol objects: {n_invalid_mol}")
        print(f"    ✗ Invalid graphs: {n_invalid_graph}")
        
        if n_valid == 0:
            raise ValueError("No valid molecules found! Check your SMILES data.")
        
        # 测试一个分子的完整流程
        if len(self.mol_graph_cache) > 0:
            test_comp_id = list(self.mol_graph_cache.keys())[0]
            test_graph = self.mol_graph_cache[test_comp_id]
            
            print(f"\n Test Molecule ({test_comp_id}):")
            print(f"    Atom features: {test_graph.V.shape}")
            print(f"    Bond features: {test_graph.E.shape}")
            print(f"    Edge index: {test_graph.edge_index.shape}")
            
            # 测试能否创建BatchMolGraph
            try:
                from chemprop.data import BatchMolGraph
                test_batch = BatchMolGraph([test_graph])
                
                if test_batch.V is None:
                    raise ValueError("BatchMolGraph.V is None after creation!")
                
                print(f"    ✓ BatchMolGraph creation: SUCCESS")
                print(f"    BatchMolGraph.V: {test_batch.V.shape}")
                
            except Exception as e:
                raise RuntimeError(f"BatchMolGraph test failed: {e}")
        
        # Create valid sample mask
        self.valid_mask = np.array([
            comp_id in self.mol_cache 
            for comp_id in self.compound_ids
        ])
        
        n_valid_samples = self.valid_mask.sum()
        n_invalid_samples = len(self.valid_mask) - n_valid_samples
        
        print(f"\n Sample Validation:")
        print(f"    Total samples: {len(self.valid_mask)}")
        print(f"    ✓ Valid samples: {n_valid_samples}")
        print(f"    ✗ Invalid samples: {n_invalid_samples}")
        
        if n_valid_samples == 0:
            raise ValueError("No valid samples found!")
    
        self.preload_to_gpu = preload_to_gpu
        
        if self.preload_to_gpu and torch.cuda.is_available():
            print("  🚀 Pre-loading molecular graphs to GPU...")
            
            gpu_cache = {}
            for comp_id, mol_graph in self.mol_graph_cache.items():
                try:
                    # 创建 GPU 版本
                    gpu_graph = type(mol_graph)(
                        V=mol_graph.V.to('cuda'),
                        E=mol_graph.E.to('cuda'),
                        edge_index=mol_graph.edge_index.to('cuda'),
                        rev_edge_index=mol_graph.rev_edge_index.to('cuda') if hasattr(mol_graph, 'rev_edge_index') else None
                    )
                    gpu_cache[comp_id] = gpu_graph
                    
                except Exception as e:
                    print(f"    !  Failed to move {comp_id} to GPU: {e}")
                    # 保留 CPU 版本
                    gpu_cache[comp_id] = mol_graph
            
            self.mol_graph_cache = gpu_cache
            print(f"  ✓ Pre-loaded {len(gpu_cache)} graphs to GPU")
        
    def __len__(self):
        return self.valid_mask.sum()
    
    def __getitem__(self, idx):
        """
        Get item by index (only valid samples).
        
        Returns
        -------
        tuple
            If use_3d=False: (x_transcript, mol_graph, y_label)
            If use_3d=True:  (x_transcript, mol_graph, coords_3d, y_label)
        """
        valid_indices = np.where(self.valid_mask)[0]
        actual_idx = valid_indices[idx]
        
        x_transcript = torch.FloatTensor(self.X[actual_idx])
        y_label = torch.LongTensor([self.y[actual_idx]])[0]
        
        comp_id = self.compound_ids[actual_idx]
        mol_graph = self.mol_graph_cache[comp_id]
        
        if mol_graph.V is None:
            raise RuntimeError(f"MolGraph.V is None for compound {comp_id} at idx {idx}")
        
        # 🔥 如果使用3D，返回坐标
        if self.use_3d:
            coords_3d = self.coords_3d_cache.get(comp_id)
            if coords_3d is None:
                # 如果没有3D坐标，返回零向量（fallback）
                n_atoms = mol_graph.V.shape[0]
                coords_3d = torch.zeros(n_atoms, 3)
            return x_transcript, mol_graph, coords_3d, y_label
        else:
            return x_transcript, mol_graph, y_label

    def _generate_conformer(self, mol):
        """
        Generate 3D conformer for a molecule.
        
        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule
        
        Returns
        -------
        np.ndarray or None
            3D coordinates (n_atoms, 3)
        """
        from conformer_utils import ConformerGenerator
        
        if not hasattr(self, '_conformer_gen'):
            self._conformer_gen = ConformerGenerator(
                method=self.conformer_method,
                num_confs=1,
                optimize=True
            )
        
        try:
            coords = self._conformer_gen.generate(mol)
            return coords
        except Exception as e:
            return None
        
def collate_two_tower(batch):
    """
    Custom collate function for batching.
    
    Parameters
    ----------
    batch : list
        List of tuples from __getitem__:
        - If use_3d=False: (x_transcript, mol_graph, y_label)
        - If use_3d=True:  (x_transcript, mol_graph, coords_3d, y_label)
    
    Returns
    -------
    tuple
        If use_3d=False: (x_batch, bmg_batch, y_batch)
        If use_3d=True:  (x_batch, bmg_batch, coords_3d_batch, y_batch)
    """
    from chemprop.data import BatchMolGraph
    
    x_transcripts = []
    mol_graphs = []
    coords_3d_list = []
    y_labels = []
    
    # 检测是否使用3D（根据batch元素长度判断）
    use_3d = len(batch[0]) == 4
    
    for item in batch:
        if use_3d:
            x_t, mol_graph, coords_3d, y = item
            coords_3d_list.append(coords_3d)
        else:
            x_t, mol_graph, y = item
        
        # 验证
        if mol_graph is None or not hasattr(mol_graph, 'V') or mol_graph.V is None:
            raise ValueError("Received None mol_graph in collate_fn")
        
        x_transcripts.append(x_t)
        mol_graphs.append(mol_graph)
        y_labels.append(y)
    
    # Stack
    x_batch = torch.stack(x_transcripts)
    y_batch = torch.stack(y_labels)
    
    # Create BatchMolGraph
    try:
        bmg_batch = BatchMolGraph(mol_graphs)
        
        if bmg_batch.V is None:
            raise ValueError("BatchMolGraph.V is None after creation")
        if bmg_batch.E is None:
            raise ValueError("BatchMolGraph.E is None after creation")
        
    except Exception as e:
        print(f"❌ Error creating BatchMolGraph: {e}")
        raise
    
    # 如果使用3D，拼接所有坐标
    if use_3d:
        coords_3d_batch = torch.cat(coords_3d_list, dim=0)  # (total_atoms, 3)
        return x_batch, bmg_batch, coords_3d_batch, y_batch
    else:
        return x_batch, bmg_batch, y_batch