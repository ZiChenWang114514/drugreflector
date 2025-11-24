#!/bin/bash
#SBATCH -J twotower-train
#SBATCH -p gpu_4l
#SBATCH -N 1
#SBATCH -o ../err_out/twotower_train_%j.out
#SBATCH -e ../err_out/twotower_train_%j.err
#SBATCH --no-requeue
#SBATCH -A tangc_g1
#SBATCH --qos=tangcg4c
#SBATCH --gres=gpu:1
#SBATCH --overcommit
#SBATCH --mincpus=30

# Activate environment
source /appsnew/source/Tensorflow2.11_Keras2.11.0_keras-nightly2.13.0_torch1.13.1_py3.9.4_cuda11.4.4.sh
unset LD_LIBRARY_PATH

# Train Two-Tower model (all folds)
python -u train.py \
    --data-file ../datasets/training_data_lincs2020_final.pkl \
    --mol-embeddings ../datasets/mol_embeddings_unimol.pkl \
    --output-dir ../models/twotower_ensemble \
    --all-folds \
    --embedding-dim 512 \
    --fusion-type concat \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 0.0139 \
    --focal-gamma 2.0 \
    --contrastive-weight 0.1 \
    --num-workers 8 \
    --save-every 10
