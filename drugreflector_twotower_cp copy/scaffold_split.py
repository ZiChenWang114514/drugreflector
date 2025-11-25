"""
Scaffold-based train/test split utilities.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, List
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(smiles: str) -> str:
    """
    Get Bemis-Murcko scaffold from SMILES.
    
    Parameters
    ----------
    smiles : str
        SMILES string
    
    Returns
    -------
    str
        Scaffold SMILES (or original SMILES if scaffold extraction fails)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return smiles


def scaffold_split(
    compound_ids: np.ndarray,
    smiles_dict: Dict[str, str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split compounds by scaffold into train/val/test sets.
    
    Ensures that compounds with the same scaffold are in the same split.
    
    Parameters
    ----------
    compound_ids : np.ndarray
        Array of compound IDs for each sample
    smiles_dict : Dict[str, str]
        Mapping from compound ID to SMILES
    train_ratio : float
        Training set ratio
    val_ratio : float
        Validation set ratio
    test_ratio : float
        Test set ratio
    seed : int
        Random seed
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Boolean masks for train, val, test
    """
    np.random.seed(seed)
    
    # Get unique compounds
    unique_compounds = np.unique(compound_ids)
    
    # Group compounds by scaffold
    scaffold_to_compounds = defaultdict(list)
    compound_to_scaffold = {}
    
    print(f"\n📊 Computing scaffolds for {len(unique_compounds)} compounds...")
    
    for comp_id in unique_compounds:
        smiles = smiles_dict.get(comp_id)
        if smiles and not pd.isna(smiles):
            scaffold = get_scaffold(smiles)
            scaffold_to_compounds[scaffold].append(comp_id)
            compound_to_scaffold[comp_id] = scaffold
        else:
            # Use compound ID as scaffold if no SMILES
            scaffold_to_compounds[comp_id].append(comp_id)
            compound_to_scaffold[comp_id] = comp_id
    
    print(f"  ✓ Found {len(scaffold_to_compounds)} unique scaffolds")
    
    # Sort scaffolds by size (number of compounds) for deterministic splitting
    scaffolds = sorted(
        scaffold_to_compounds.keys(),
        key=lambda x: len(scaffold_to_compounds[x]),
        reverse=True
    )
    
    # Shuffle scaffolds
    np.random.shuffle(scaffolds)
    
    # Assign scaffolds to splits
    n_total = len(compound_ids)
    train_cutoff = int(train_ratio * n_total)
    val_cutoff = int((train_ratio + val_ratio) * n_total)
    
    train_compounds = set()
    val_compounds = set()
    test_compounds = set()
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    for scaffold in scaffolds:
        compounds = scaffold_to_compounds[scaffold]
        # Count samples for this scaffold
        scaffold_sample_count = sum(
            (compound_ids == comp).sum() for comp in compounds
        )
        
        # Assign to split based on current counts
        if train_count + scaffold_sample_count <= train_cutoff:
            train_compounds.update(compounds)
            train_count += scaffold_sample_count
        elif val_count + scaffold_sample_count <= val_cutoff - train_cutoff:
            val_compounds.update(compounds)
            val_count += scaffold_sample_count
        else:
            test_compounds.update(compounds)
            test_count += scaffold_sample_count
    
    # Create masks
    train_mask = np.array([comp in train_compounds for comp in compound_ids])
    val_mask = np.array([comp in val_compounds for comp in compound_ids])
    test_mask = np.array([comp in test_compounds for comp in compound_ids])
    
    # Statistics
    print(f"\n📊 Scaffold Split Statistics:")
    print(f"  Train: {train_mask.sum():,} samples ({train_mask.sum()/len(compound_ids)*100:.1f}%)")
    print(f"    - {len(train_compounds)} compounds")
    print(f"  Val: {val_mask.sum():,} samples ({val_mask.sum()/len(compound_ids)*100:.1f}%)")
    print(f"    - {len(val_compounds)} compounds")
    print(f"  Test: {test_mask.sum():,} samples ({test_mask.sum()/len(compound_ids)*100:.1f}%)")
    print(f"    - {len(test_compounds)} compounds")
    
    # Check for scaffold overlap
    train_scaffolds = set(compound_to_scaffold.get(c) for c in train_compounds if c in compound_to_scaffold)
    val_scaffolds = set(compound_to_scaffold.get(c) for c in val_compounds if c in compound_to_scaffold)
    test_scaffolds = set(compound_to_scaffold.get(c) for c in test_compounds if c in compound_to_scaffold)
    
    overlap_train_val = train_scaffolds & val_scaffolds
    overlap_train_test = train_scaffolds & test_scaffolds
    overlap_val_test = val_scaffolds & test_scaffolds
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print(f"  ⚠️ Warning: Scaffold overlap detected!")
        if overlap_train_val:
            print(f"    Train-Val: {len(overlap_train_val)} scaffolds")
        if overlap_train_test:
            print(f"    Train-Test: {len(overlap_train_test)} scaffolds")
        if overlap_val_test:
            print(f"    Val-Test: {len(overlap_val_test)} scaffolds")
    else:
        print(f"  ✓ No scaffold overlap between splits")
    
    return train_mask, val_mask, test_mask


def create_scaffold_folds(
    compound_ids: np.ndarray,
    smiles_dict: Dict[str, str],
    n_folds: int = 3,
    seed: int = 42
) -> np.ndarray:
    """
    Create scaffold-based k-fold splits.
    
    Parameters
    ----------
    compound_ids : np.ndarray
        Array of compound IDs for each sample
    smiles_dict : Dict[str, str]
        Mapping from compound ID to SMILES
    n_folds : int
        Number of folds
    seed : int
        Random seed
    
    Returns
    -------
    np.ndarray
        Fold assignments (0 to n_folds-1)
    """
    np.random.seed(seed)
    
    # Get unique compounds and scaffolds
    unique_compounds = np.unique(compound_ids)
    
    scaffold_to_compounds = defaultdict(list)
    compound_to_scaffold = {}
    
    print(f"\n📊 Computing scaffolds for {len(unique_compounds)} compounds...")
    
    for comp_id in unique_compounds:
        smiles = smiles_dict.get(comp_id)
        if smiles and not pd.isna(smiles):
            scaffold = get_scaffold(smiles)
            scaffold_to_compounds[scaffold].append(comp_id)
            compound_to_scaffold[comp_id] = scaffold
        else:
            scaffold_to_compounds[comp_id].append(comp_id)
            compound_to_scaffold[comp_id] = comp_id
    
    print(f"  ✓ Found {len(scaffold_to_compounds)} unique scaffolds")
    
    # Sort and shuffle scaffolds
    scaffolds = sorted(
        scaffold_to_compounds.keys(),
        key=lambda x: len(scaffold_to_compounds[x]),
        reverse=True
    )
    np.random.shuffle(scaffolds)
    
    # Assign scaffolds to folds (round-robin style)
    fold_counts = [0] * n_folds
    scaffold_to_fold = {}
    
    for scaffold in scaffolds:
        compounds = scaffold_to_compounds[scaffold]
        scaffold_sample_count = sum(
            (compound_ids == comp).sum() for comp in compounds
        )
        
        # Assign to fold with smallest count
        target_fold = np.argmin(fold_counts)
        scaffold_to_fold[scaffold] = target_fold
        fold_counts[target_fold] += scaffold_sample_count
    
    # Create fold array
    folds = np.zeros(len(compound_ids), dtype=int)
    for i, comp_id in enumerate(compound_ids):
        scaffold = compound_to_scaffold.get(comp_id, comp_id)
        folds[i] = scaffold_to_fold.get(scaffold, 0)
    
    # Statistics
    print(f"\n📊 Scaffold-based {n_folds}-Fold Statistics:")
    for fold_id in range(n_folds):
        fold_mask = folds == fold_id
        fold_compounds = set(compound_ids[fold_mask])
        print(f"  Fold {fold_id}: {fold_mask.sum():,} samples ({fold_mask.sum()/len(compound_ids)*100:.1f}%)")
        print(f"    - {len(fold_compounds)} compounds")
    
    return folds

def cold_start_split(
    compound_ids: np.ndarray,
    smiles_dict: Dict[str, str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    cold_start_ratio: float = 0.3,  # 30%的化合物为cold start
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cold start split: 结合random和scaffold split的优点.
    
    策略:
    - cold_start_ratio的化合物完全不出现在训练集(类似scaffold split)
    - 其余化合物可以在train/val/test中共享(类似random split)
    
    Parameters
    ----------
    compound_ids : np.ndarray
        Array of compound IDs for each sample
    smiles_dict : Dict[str, str]
        Mapping from compound ID to SMILES
    train_ratio : float
        Training set ratio
    val_ratio : float
        Validation set ratio
    test_ratio : float
        Test set ratio
    cold_start_ratio : float
        Proportion of compounds to reserve as cold start (unseen in training)
    seed : int
        Random seed
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Boolean masks for train, val, test
    """
    np.random.seed(seed)
    
    # Get unique compounds and group by scaffold
    unique_compounds = np.unique(compound_ids)
    
    scaffold_to_compounds = defaultdict(list)
    compound_to_scaffold = {}
    
    print(f"\n📊 Cold Start Split (cold_start_ratio={cold_start_ratio})...")
    print(f"  Computing scaffolds for {len(unique_compounds)} compounds...")
    
    for comp_id in unique_compounds:
        smiles = smiles_dict.get(comp_id)
        if smiles and not pd.isna(smiles):
            scaffold = get_scaffold(smiles)
            scaffold_to_compounds[scaffold].append(comp_id)
            compound_to_scaffold[comp_id] = scaffold
        else:
            scaffold_to_compounds[comp_id].append(comp_id)
            compound_to_scaffold[comp_id] = comp_id
    
    print(f"  ✓ Found {len(scaffold_to_compounds)} unique scaffolds")
    
    # Sort scaffolds by size
    scaffolds = sorted(
        scaffold_to_compounds.keys(),
        key=lambda x: len(scaffold_to_compounds[x]),
        reverse=True
    )
    np.random.shuffle(scaffolds)
    
    # Step 1: 选择cold start scaffolds (基于scaffold)
    n_cold_start_scaffolds = int(len(scaffolds) * cold_start_ratio)
    cold_start_scaffolds = set(scaffolds[:n_cold_start_scaffolds])
    warm_start_scaffolds = set(scaffolds[n_cold_start_scaffolds:])
    
    cold_start_compounds = set()
    for scaffold in cold_start_scaffolds:
        cold_start_compounds.update(scaffold_to_compounds[scaffold])
    
    warm_start_compounds = set()
    for scaffold in warm_start_scaffolds:
        warm_start_compounds.update(scaffold_to_compounds[scaffold])
    
    print(f"\n  Split strategy:")
    print(f"    Cold start compounds: {len(cold_start_compounds)} ({len(cold_start_compounds)/len(unique_compounds)*100:.1f}%)")
    print(f"    Warm start compounds: {len(warm_start_compounds)} ({len(warm_start_compounds)/len(unique_compounds)*100:.1f}%)")
    
    # Step 2: 将cold start化合物的样本分配到val和test
    cold_start_sample_mask = np.isin(compound_ids, list(cold_start_compounds))
    cold_start_indices = np.where(cold_start_sample_mask)[0]
    np.random.shuffle(cold_start_indices)
    
    n_cold_samples = len(cold_start_indices)
    n_cold_val = int(n_cold_samples * 0.5)  # 50-50 split between val and test
    
    cold_val_indices = cold_start_indices[:n_cold_val]
    cold_test_indices = cold_start_indices[n_cold_val:]
    
    # Step 3: 将warm start化合物的样本随机分配到train/val/test
    warm_start_sample_mask = np.isin(compound_ids, list(warm_start_compounds))
    warm_start_indices = np.where(warm_start_sample_mask)[0]
    np.random.shuffle(warm_start_indices)
    
    n_warm_samples = len(warm_start_indices)
    n_warm_train = int(n_warm_samples * train_ratio)
    n_warm_val = int(n_warm_samples * val_ratio)
    
    warm_train_indices = warm_start_indices[:n_warm_train]
    warm_val_indices = warm_start_indices[n_warm_train:n_warm_train + n_warm_val]
    warm_test_indices = warm_start_indices[n_warm_train + n_warm_val:]
    
    # Step 4: 组合masks
    train_mask = np.zeros(len(compound_ids), dtype=bool)
    val_mask = np.zeros(len(compound_ids), dtype=bool)
    test_mask = np.zeros(len(compound_ids), dtype=bool)
    
    train_mask[warm_train_indices] = True
    val_mask[np.concatenate([cold_val_indices, warm_val_indices])] = True
    test_mask[np.concatenate([cold_test_indices, warm_test_indices])] = True
    
    # Statistics
    print(f"\n📊 Cold Start Split Statistics:")
    print(f"  Train: {train_mask.sum():,} samples ({train_mask.sum()/len(compound_ids)*100:.1f}%)")
    train_compounds = set(compound_ids[train_mask])
    print(f"    - {len(train_compounds)} compounds (all warm start)")
    
    print(f"  Val: {val_mask.sum():,} samples ({val_mask.sum()/len(compound_ids)*100:.1f}%)")
    val_compounds = set(compound_ids[val_mask])
    val_cold = val_compounds & cold_start_compounds
    val_warm = val_compounds & warm_start_compounds
    print(f"    - {len(val_cold)} cold start compounds")
    print(f"    - {len(val_warm)} warm start compounds")
    
    print(f"  Test: {test_mask.sum():,} samples ({test_mask.sum()/len(compound_ids)*100:.1f}%)")
    test_compounds = set(compound_ids[test_mask])
    test_cold = test_compounds & cold_start_compounds
    test_warm = test_compounds & warm_start_compounds
    print(f"    - {len(test_cold)} cold start compounds")
    print(f"    - {len(test_warm)} warm start compounds")
    
    # Verify no cold start compounds in training
    overlap = train_compounds & cold_start_compounds
    if len(overlap) > 0:
        raise ValueError(f"ERROR: {len(overlap)} cold start compounds found in training set!")
    
    print(f"  ✓ No cold start compounds in training set (verified)")
    
    return train_mask, val_mask, test_mask


def create_cold_start_folds(
    compound_ids: np.ndarray,
    smiles_dict: Dict[str, str],
    n_folds: int = 3,
    cold_start_ratio: float = 0.3,
    seed: int = 42
) -> np.ndarray:
    """
    Create cold start k-fold splits.
    
    Parameters
    ----------
    compound_ids : np.ndarray
        Array of compound IDs for each sample
    smiles_dict : Dict[str, str]
        Mapping from compound ID to SMILES
    n_folds : int
        Number of folds
    cold_start_ratio : float
        Proportion of compounds to reserve as cold start per fold
    seed : int
        Random seed
    
    Returns
    -------
    np.ndarray
        Fold assignments (0 to n_folds-1)
    """
    np.random.seed(seed)
    
    # Get unique compounds and scaffolds
    unique_compounds = np.unique(compound_ids)
    
    scaffold_to_compounds = defaultdict(list)
    compound_to_scaffold = {}
    
    print(f"\n📊 Creating Cold Start {n_folds}-Fold CV (cold_start_ratio={cold_start_ratio})...")
    print(f"  Computing scaffolds for {len(unique_compounds)} compounds...")
    
    for comp_id in unique_compounds:
        smiles = smiles_dict.get(comp_id)
        if smiles and not pd.isna(smiles):
            scaffold = get_scaffold(smiles)
            scaffold_to_compounds[scaffold].append(comp_id)
            compound_to_scaffold[comp_id] = scaffold
        else:
            scaffold_to_compounds[comp_id].append(comp_id)
            compound_to_scaffold[comp_id] = comp_id
    
    print(f"  ✓ Found {len(scaffold_to_compounds)} unique scaffolds")
    
    # Assign scaffolds to cold/warm groups
    scaffolds = list(scaffold_to_compounds.keys())
    np.random.shuffle(scaffolds)
    
    n_cold_scaffolds_per_fold = int(len(scaffolds) * cold_start_ratio / n_folds)
    
    # Create fold assignments
    folds = np.zeros(len(compound_ids), dtype=int)
    
    # Assign cold start scaffolds to specific folds
    for fold_id in range(n_folds):
        start_idx = fold_id * n_cold_scaffolds_per_fold
        end_idx = start_idx + n_cold_scaffolds_per_fold
        cold_scaffolds_for_fold = scaffolds[start_idx:end_idx]
        
        for scaffold in cold_scaffolds_for_fold:
            compounds = scaffold_to_compounds[scaffold]
            for comp in compounds:
                comp_mask = compound_ids == comp
                folds[comp_mask] = fold_id
    
    # Assign remaining warm start scaffolds randomly
    cold_scaffolds = set(scaffolds[:n_folds * n_cold_scaffolds_per_fold])
    warm_scaffolds = set(scaffolds[n_folds * n_cold_scaffolds_per_fold:])
    
    for scaffold in warm_scaffolds:
        compounds = scaffold_to_compounds[scaffold]
        for comp in compounds:
            comp_mask = compound_ids == comp
            # Randomly assign samples of warm start compounds to folds
            warm_sample_indices = np.where(comp_mask)[0]
            warm_folds = np.random.randint(0, n_folds, len(warm_sample_indices))
            folds[warm_sample_indices] = warm_folds
    
    # Statistics
    print(f"\n📊 Cold Start {n_folds}-Fold Statistics:")
    for fold_id in range(n_folds):
        fold_mask = folds == fold_id
        fold_compounds = set(compound_ids[fold_mask])
        print(f"  Fold {fold_id}: {fold_mask.sum():,} samples ({fold_mask.sum()/len(compound_ids)*100:.1f}%)")
        print(f"    - {len(fold_compounds)} compounds")
    
    return folds