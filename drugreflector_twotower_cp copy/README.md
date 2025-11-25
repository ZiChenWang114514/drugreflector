# 使用cold start split训练
python train.py \
    --data-file training_data_lincs2020_final.pkl \
    --compound-info compoundinfo_beta.txt \
    --output-dir output/cold_start_30pct \
    --use-cold-start-split \
    --cold-start-ratio 0.3 \
    --cold-start-folds 3 \
    --all-folds

# 调整cold start比例（更严格）
python train.py \
    --data-file training_data_lincs2020_final.pkl \
    --compound-info compoundinfo_beta.txt \
    --output-dir output/cold_start_50pct \
    --use-cold-start-split \
    --cold-start-ratio 0.5 \
    --all-folds