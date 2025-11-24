#!/bin/bash
#SBATCH -J twotower-precompute
#SBATCH -p gpu_4l
#SBATCH -N 1
#SBATCH -o ../err_out/twotower_precompute_%j.out
#SBATCH -e ../err_out/twotower_precompute_%j.err
#SBATCH --no-requeue
#SBATCH -A tangc_g1
#SBATCH --qos=tangcg4c
#SBATCH --gres=gpu:1
#SBATCH --overcommit
#SBATCH --mincpus=30

# Activate environment
source /appsnew/source/Tensorflow2.11_Keras2.11.0_keras-nightly2.13.0_torch1.13.1_py3.9.4_cuda11.4.4.sh
unset LD_LIBRARY_PATH

# Step 1: Precompute molecular embeddings
# This only needs to be run once

# IMPORTANT: Set the path to your pre-downloaded Uni-Mol weights
# Download weights first on a machine with internet:
#   python download_unimol_weights.py --output-dir ../unimol_weights
# Then transfer to HPC

WEIGHTS_DIR="../unimol_weights"

python -u precompute_mol_embeddings.py \
    --compound-info ../datasets/LINCS2020/compoundinfo_beta.txt \
    --training-data ../datasets/training_data_lincs2020_final.pkl \
    --output ../datasets/mol_embeddings_unimol.pkl \
    --batch-size 64 \
    --device cuda \
    --weights-dir ${WEIGHTS_DIR}

# If weights are not available, use fallback (RDKit fingerprints):
# python -u precompute_mol_embeddings.py \
#     --compound-info ../datasets/LINCS2020/compoundinfo_beta.txt \
#     --training-data ../datasets/training_data_lincs2020_final.pkl \
#     --output ../datasets/mol_embeddings_rdkit.pkl \
#     --batch-size 64 \
#     --device cuda \
#     --use-fallback