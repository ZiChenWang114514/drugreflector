"""
3D Conformer Generation Utilities

Provides tools to generate 3D molecular conformations from SMILES,
which can be integrated with 2D graph representations.
"""
import numpy as np
import torch
from typing import Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class ConformerGenerator:
    """
    Generate low-energy 3D conformers from SMILES.
    
    Uses RDKit's ETKDG (Extended Experimental Torsion-knowledge Distance Geometry)
    algorithm for fast, high-quality conformer generation.
    
    Parameters
    ----------
    method : str
        Conformer generation method: 'ETKDG', 'MMFF', or 'UFF'
    num_confs : int
        Number of conformers to generate (will select lowest energy)
    optimize : bool
        Whether to optimize with force field
    """
    
    def __init__(
        self,
        method: str = 'ETKDG',
        num_confs: int = 1,
        optimize: bool = True
    ):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for 3D conformer generation")
        
        self.method = method
        self.num_confs = num_confs
        self.optimize = optimize
    
    def generate(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """
        Generate 3D conformer for molecule.
        
        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object
        
        Returns
        -------
        np.ndarray or None
            Atomic coordinates (n_atoms, 3), or None if failed
        """
        try:
            # Add hydrogens (important for accurate 3D geometry)
            mol = Chem.AddHs(mol)
            
            # Generate conformer(s)
            if self.method == 'ETKDG':
                # ETKDG: Best for drug-like molecules
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                conf_ids = AllChem.EmbedMultipleConfs(
                    mol, 
                    numConfs=self.num_confs,
                    params=params
                )
            else:
                # Fallback to basic embedding
                conf_ids = AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=self.num_confs,
                    randomSeed=42
                )
            
            if len(conf_ids) == 0:
                return None
            
            # Optimize geometry
            if self.optimize:
                if self.method == 'MMFF':
                    # MMFF94: Accurate but slower
                    for conf_id in conf_ids:
                        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
                else:
                    # UFF: Fast universal force field
                    for conf_id in conf_ids:
                        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
            
            # Select lowest energy conformer
            if self.num_confs > 1:
                energies = []
                for conf_id in conf_ids:
                    if self.method == 'MMFF':
                        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id)
                        energies.append(ff.CalcEnergy())
                    else:
                        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                        energies.append(ff.CalcEnergy())
                
                best_conf_id = conf_ids[np.argmin(energies)]
            else:
                best_conf_id = conf_ids[0]
            
            # Extract coordinates
            conf = mol.GetConformer(best_conf_id)
            coords = conf.GetPositions()  # (n_atoms, 3)
            
            # Remove hydrogens for consistency with 2D graphs
            # (Chemprop typically doesn't include explicit H)
            mol_no_h = Chem.RemoveHs(mol)
            n_heavy_atoms = mol_no_h.GetNumAtoms()
            
            # Map heavy atom coordinates
            heavy_coords = []
            h_count = 0
            for i, atom in enumerate(mol.GetAtoms()):
                if atom.GetAtomicNum() > 1:  # Not hydrogen
                    heavy_coords.append(coords[i])
                    h_count += 1
                    if h_count >= n_heavy_atoms:
                        break
            
            return np.array(heavy_coords, dtype=np.float32)
        
        except Exception as e:
            print(f"⚠️  Conformer generation failed: {e}")
            return None
    
    def generate_from_smiles(self, smiles: str) -> Optional[np.ndarray]:
        """
        Generate conformer directly from SMILES string.
        
        Parameters
        ----------
        smiles : str
            SMILES string
        
        Returns
        -------
        np.ndarray or None
            Atomic coordinates (n_atoms, 3)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return self.generate(mol)
        except:
            return None


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix from 3D coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        Atomic coordinates (n_atoms, 3)
    
    Returns
    -------
    np.ndarray
        Distance matrix (n_atoms, n_atoms)
    """
    n_atoms = coords.shape[0]
    dist_matrix = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


def compute_geometric_features(coords: np.ndarray) -> dict:
    """
    Compute rich geometric features from 3D coordinates.
    
    Features include:
    - Distance to centroid
    - Principal moments (shape descriptors)
    - Local geometry (angles, dihedrals)
    
    Parameters
    ----------
    coords : np.ndarray
        Atomic coordinates (n_atoms, 3)
    
    Returns
    -------
    dict
        Dictionary of geometric features
    """
    # Centroid
    centroid = coords.mean(axis=0)
    
    # Distance to centroid
    dist_to_center = np.linalg.norm(coords - centroid, axis=1)
    
    # Radius of gyration
    rg = np.sqrt(np.mean(dist_to_center ** 2))
    
    # Principal moments (via PCA)
    centered = coords - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    
    # Shape descriptors from moments
    asphericity = eigenvalues[0] - 0.5 * (eigenvalues[1] + eigenvalues[2])
    acylindricity = eigenvalues[1] - eigenvalues[2]
    
    # Distance matrix
    dist_matrix = compute_distance_matrix(coords)
    
    return {
        'coords': coords,
        'centroid': centroid,
        'dist_to_center': dist_to_center,
        'radius_of_gyration': rg,
        'eigenvalues': eigenvalues,
        'asphericity': asphericity,
        'acylindricity': acylindricity,
        'dist_matrix': dist_matrix
    }


def coords_to_atom_features(coords: np.ndarray, feature_dim: int = 16) -> np.ndarray:
    """
    Convert 3D coordinates to per-atom feature vectors.
    
    This creates a compact representation suitable for appending to
    existing atom features in Chemprop.
    
    Parameters
    ----------
    coords : np.ndarray
        Atomic coordinates (n_atoms, 3)
    feature_dim : int
        Output feature dimension per atom
    
    Returns
    -------
    np.ndarray
        Atom-level 3D features (n_atoms, feature_dim)
    """
    geom = compute_geometric_features(coords)
    
    n_atoms = coords.shape[0]
    features = np.zeros((n_atoms, feature_dim), dtype=np.float32)
    
    # Normalized coordinates (first 3 dims)
    coords_normalized = (coords - geom['centroid']) / (geom['radius_of_gyration'] + 1e-8)
    features[:, 0:3] = coords_normalized
    
    # Distance to centroid (normalized)
    dist_norm = geom['dist_to_center'] / (geom['dist_to_center'].max() + 1e-8)
    features[:, 3] = dist_norm
    
    # Mean distance to other atoms
    mean_dist = geom['dist_matrix'].mean(axis=1)
    mean_dist_norm = mean_dist / (mean_dist.max() + 1e-8)
    features[:, 4] = mean_dist_norm
    
    # Min/max distances
    min_dist = np.array([geom['dist_matrix'][i, geom['dist_matrix'][i] > 0].min() 
                        for i in range(n_atoms)])
    max_dist = geom['dist_matrix'].max(axis=1)
    features[:, 5] = min_dist / (max_dist.max() + 1e-8)
    features[:, 6] = max_dist / (max_dist.max() + 1e-8)
    
    # Global shape descriptors (broadcast to all atoms)
    features[:, 7] = geom['asphericity']
    features[:, 8] = geom['acylindricity']
    features[:, 9:12] = geom['eigenvalues']  # 3 principal moments
    
    # Remaining dims: padding or additional features
    # Could add: local curvature, surface area contributions, etc.
    
    return features


# Example usage
if __name__ == "__main__":
    # Test conformer generation
    gen = ConformerGenerator(method='ETKDG', num_confs=5)
    
    smiles = "CCO"  # Ethanol
    coords = gen.generate_from_smiles(smiles)
    
    if coords is not None:
        print(f"Generated 3D coordinates for {smiles}")
        print(f"Shape: {coords.shape}")
        print(f"Coordinates:\n{coords}")
        
        # Compute features
        features = coords_to_atom_features(coords, feature_dim=16)
        print(f"\n3D Features shape: {features.shape}")
        print(f"Features (first atom):\n{features[0]}")