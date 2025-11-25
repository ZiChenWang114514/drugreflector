"""
Scaffold-based train/test split utilities for Two-Tower DrugReflector.

Based on Bemis-Murcko scaffold splitting to ensure compounds with similar
core structures are not present in both training and test sets.
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
    sample_meta: pd.DataFrame,
    smiles_dict: Dict[str, str],
    pert_id_col: str = 'pert_id',
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
    sample_meta : pd.DataFrame
        Sample metadata containing compound IDs
    smiles_dict : Dict[str, str]
        Mapping from compound ID to SMILES
    pert_id_col : str
        Column name for compound IDs
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
    
    # Get compound IDs for each sample
    compound_ids = sample_meta[pert_id_col].values
    n_samples = len(compound_ids)
    
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
    train_cutoff = int(train_ratio * n_samples)
    val_cutoff = int((train_ratio + val_ratio) * n_samples)
    
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
    print(f"  Train: {train_mask.sum():,} samples ({train_mask.sum()/n_samples*100:.1f}%)")
    print(f"    - {len(train_compounds)} compounds")
    print(f"  Val: {val_mask.sum():,} samples ({val_mask.sum()/n_samples*100:.1f}%)")
    print(f"    - {len(val_compounds)} compounds")
    print(f"  Test: {test_mask.sum():,} samples ({test_mask.sum()/n_samples*100:.1f}%)")
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
    sample_meta: pd.DataFrame,
    smiles_dict: Dict[str, str],
    pert_id_col: str = 'pert_id',
    n_folds: int = 3,
    seed: int = 42
) -> np.ndarray:
    """
    Create scaffold-based k-fold splits.
    
    Parameters
    ----------
    sample_meta : pd.DataFrame
        Sample metadata containing compound IDs
    smiles_dict : Dict[str, str]
        Mapping from compound ID to SMILES
    pert_id_col : str
        Column name for compound IDs
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
    
    # Get compound IDs for each sample
    compound_ids = sample_meta[pert_id_col].values
    n_samples = len(compound_ids)
    
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
    folds = np.zeros(n_samples, dtype=int)
    for i, comp_id in enumerate(compound_ids):
        scaffold = compound_to_scaffold.get(comp_id, comp_id)
        folds[i] = scaffold_to_fold.get(scaffold, 0)
    
    # Statistics
    print(f"\n📊 Scaffold-based {n_folds}-Fold Statistics:")
    for fold_id in range(n_folds):
        fold_mask = folds == fold_id
        fold_compounds = set(compound_ids[fold_mask])
        print(f"  Fold {fold_id}: {fold_mask.sum():,} samples ({fold_mask.sum()/n_samples*100:.1f}%)")
        print(f"    - {len(fold_compounds)} compounds")
    
    return folds