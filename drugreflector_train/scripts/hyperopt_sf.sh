#!/bin/bash
#SBATCH -J drugreflector-hyperparam-1fold
#SBATCH -p gpu_4l
#SBATCH -N 1 
#SBATCH -o ../err_out/hyperparam_1fold_%A_%a.out
#SBATCH -e ../err_out/hyperparam_1fold_%A_%a.err
#SBATCH --no-requeue
#SBATCH -A tangc_g1
#SBATCH --qos=tangcg4c
#SBATCH --gres=gpu:1
#SBATCH --overcommit
#SBATCH --mincpus=30
#SBATCH --array=0-215%12  # 216个配置 (6 LR × 3 scheduler × 3 WD × 4 dropout)，最多同时12个

# ============================================================================
# DrugReflector 单折超参数优化脚本
# 每个任务只训练一个 fold (fold 0)，加速超参数搜索
# ============================================================================

source /appsnew/source/Miniconda3-py38_4.10.3.sh
conda activate env_chemprop_4l

# ============================================================================
# 配置区域
# ============================================================================

# 基础路径
DATA_FILE="../datasets/training_data_lincs2020_final.pkl"
BASE_OUTPUT_DIR="../models/hyperparam_search_1fold"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 训练基础参数
EPOCHS=50
BATCH_SIZE=256
DEVICE="auto"
NUM_WORKERS=30
PLOT_DIR="training_plots"

# 固定使用 fold 0 进行超参数搜索
FOLD_ID=0

# ============================================================================
# 超参数搜索空间定义
# ============================================================================

# 学习率搜索空间
LR_VALUES=(0.001 0.005 0.01 0.0139 0.02 0.05)

# 学习率调度器类型
LR_SCHEDULER_VALUES=("step" "exponential" "cosine")

# 权重衰减 (L2 正则化)
WEIGHT_DECAY_VALUES=(1e-6 1e-5 1e-4)

# Dropout rate (新增)
# 参考: 原始论文使用 0.64
DROPOUT_VALUES=(0.3 0.5 0.64 0.7)

# Focal Loss gamma
FOCAL_GAMMA=2.0

# ============================================================================
# 超参数组合生成
# ============================================================================

# 计算总配置数
N_LR=${#LR_VALUES[@]}
N_SCHEDULER=${#LR_SCHEDULER_VALUES[@]}
N_WD=${#WEIGHT_DECAY_VALUES[@]}
N_DROPOUT=${#DROPOUT_VALUES[@]}
TOTAL_CONFIGS=$((N_LR * N_SCHEDULER * N_WD * N_DROPOUT))

echo "=================================================="
echo "单折超参数优化任务开始"
echo "=================================================="
echo "任务阵列 ID: ${SLURM_ARRAY_TASK_ID}"
echo "任务总数: ${TOTAL_CONFIGS}"
echo "训练 Fold: ${FOLD_ID}"
echo "时间戳: ${TIMESTAMP}"
echo "数据文件: ${DATA_FILE}"
echo "=================================================="

# 将任务ID转换为超参数组合
TASK_ID=${SLURM_ARRAY_TASK_ID}

# 使用4D索引解码
idx_lr=$((TASK_ID % N_LR))
idx_scheduler=$(((TASK_ID / N_LR) % N_SCHEDULER))
idx_wd=$(((TASK_ID / (N_LR * N_SCHEDULER)) % N_WD))
idx_dropout=$((TASK_ID / (N_LR * N_SCHEDULER * N_WD)))

# 获取当前任务的超参数
CURRENT_LR=${LR_VALUES[$idx_lr]}
CURRENT_SCHEDULER=${LR_SCHEDULER_VALUES[$idx_scheduler]}
CURRENT_WD=${WEIGHT_DECAY_VALUES[$idx_wd]}
CURRENT_DROPOUT=${DROPOUT_VALUES[$idx_dropout]}

# 固定其他超参数
CURRENT_FOCAL_GAMMA=${FOCAL_GAMMA}
CURRENT_MIN_LR=1e-6

# 根据调度器类型设置衰减参数
if [ "$CURRENT_SCHEDULER" = "step" ]; then
    LR_DECAY_RATE=0.1
    LR_DECAY_EPOCHS="30 40"
elif [ "$CURRENT_SCHEDULER" = "exponential" ]; then
    LR_DECAY_RATE=0.95
    LR_DECAY_EPOCHS=""
else  # cosine
    LR_DECAY_RATE=0.1
    LR_DECAY_EPOCHS=""
fi

# 创建配置标识符
CONFIG_ID="lr${CURRENT_LR}_sch${CURRENT_SCHEDULER}_wd${CURRENT_WD}_drop${CURRENT_DROPOUT}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/config_${CONFIG_ID}_task${SLURM_ARRAY_TASK_ID}"

# ============================================================================
# 打印当前配置
# ============================================================================
echo ""
echo "当前配置 (任务 ID: ${TASK_ID}):"
echo "  学习率: ${CURRENT_LR}"
echo "  调度器: ${CURRENT_SCHEDULER}"
echo "  权重衰减: ${CURRENT_WD}"
echo "  Dropout: ${CURRENT_DROPOUT}"
echo "  Focal Gamma: ${CURRENT_FOCAL_GAMMA}"
echo "  最小学习率: ${CURRENT_MIN_LR}"
if [ -n "$LR_DECAY_EPOCHS" ]; then
    echo "  衰减轮次: ${LR_DECAY_EPOCHS}"
fi
echo "  训练 Fold: ${FOLD_ID}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "=================================================="
echo ""

# ============================================================================
# 执行训练
# ============================================================================

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 保存配置到文件
cat > "${OUTPUT_DIR}/config.txt" << EOF
================================================
超参数配置 - 任务 ${SLURM_ARRAY_TASK_ID}
================================================
训练参数:
  数据文件: ${DATA_FILE}
  训练 Fold: ${FOLD_ID}
  训练轮次: ${EPOCHS}
  批次大小: ${BATCH_SIZE}
  设备: ${DEVICE}
  工作进程: ${NUM_WORKERS}

优化超参数:
  学习率: ${CURRENT_LR}
  学习率调度器: ${CURRENT_SCHEDULER}
  学习率衰减率: ${LR_DECAY_RATE}
  学习率衰减轮次: ${LR_DECAY_EPOCHS}
  最小学习率: ${CURRENT_MIN_LR}
  权重衰减: ${CURRENT_WD}
  Dropout: ${CURRENT_DROPOUT}
  Focal Gamma: ${CURRENT_FOCAL_GAMMA}

任务信息:
  SLURM Job ID: ${SLURM_JOB_ID}
  SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}
  节点: ${SLURM_NODELIST}
  开始时间: $(date)
================================================
EOF

# 构建训练命令 - 使用 --fold 而不是 --all-folds
CMD="python -u ../drugreflector_train/train.py \
    --data-file ${DATA_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --fold ${FOLD_ID} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${CURRENT_LR} \
    --lr-scheduler ${CURRENT_SCHEDULER} \
    --lr-decay-rate ${LR_DECAY_RATE} \
    --min-lr ${CURRENT_MIN_LR} \
    --weight-decay ${CURRENT_WD} \
    --dropout ${CURRENT_DROPOUT} \
    --focal-gamma ${CURRENT_FOCAL_GAMMA} \
    --plot-dir ${PLOT_DIR} \
    --device ${DEVICE} \
    --num-workers ${NUM_WORKERS} \
    --save-every 10"

# 添加衰减轮次参数（仅对 step scheduler）
if [ -n "$LR_DECAY_EPOCHS" ]; then
    CMD="${CMD} --lr-decay-epochs ${LR_DECAY_EPOCHS}"
fi

echo "执行命令:"
echo "${CMD}"
echo ""
echo "开始训练..."
echo "=================================================="

# 记录开始时间
START_TIME=$(date +%s)

# 执行训练
eval ${CMD}

# 检查训练是否成功
TRAIN_STATUS=$?

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED_TIME / 3600))
ELAPSED_MINS=$(((ELAPSED_TIME % 3600) / 60))
ELAPSED_SECS=$((ELAPSED_TIME % 60))

echo ""
echo "=================================================="
echo "训练完成 - 任务 ${SLURM_ARRAY_TASK_ID}"
echo "=================================================="
echo "  状态码: ${TRAIN_STATUS}"
echo "  用时: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m ${ELAPSED_SECS}s"
echo "  输出目录: ${OUTPUT_DIR}"
echo "=================================================="

# 记录完成信息
cat >> "${OUTPUT_DIR}/config.txt" << EOF

结果:
  训练状态: $(if [ $TRAIN_STATUS -eq 0 ]; then echo "成功"; else echo "失败 (${TRAIN_STATUS})"; fi)
  完成时间: $(date)
  运行时长: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m ${ELAPSED_SECS}s
================================================
EOF

# ============================================================================
# 提取和保存关键指标（单折版本）
# ============================================================================

if [ $TRAIN_STATUS -eq 0 ]; then
    echo "提取训练指标..."
    
    # 使用 Python 提取最佳指标
    python -u << 'PYTHON_SCRIPT'
import pickle
import sys
from pathlib import Path

try:
    output_dir = Path("'${OUTPUT_DIR}'")
    
    # 单折训练不会有 ensemble_summary.pkl，需要从模型 checkpoint 中提取
    model_file = output_dir / f"model_fold_{int('${FOLD_ID}')}.pt"
    
    if model_file.exists():
        checkpoint = torch.load(model_file, map_location='cpu')
        history = checkpoint['history']
        
        # 获取最佳指标
        best_recall_idx = max(range(len(history['val_recall'])), 
                             key=lambda i: history['val_recall'][i])
        best_recall = history['val_recall'][best_recall_idx]
        best_top1_acc = history['val_top1_acc'][best_recall_idx]
        best_top10_acc = history['val_top10_acc'][best_recall_idx]
        
        # 保存简洁的结果摘要
        results_file = output_dir / "results_summary.txt"
        with open(results_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"单折训练结果摘要 (Fold ${FOLD_ID})\n")
            f.write("="*60 + "\n\n")
            f.write(f"最佳指标 (Epoch {best_recall_idx + 1}):\n")
            f.write(f"  Top 1% Recall: {best_recall:.4f}\n")
            f.write(f"  Top-1 Acc: {best_top1_acc:.4f}\n")
            f.write(f"  Top-10 Acc: {best_top10_acc:.4f}\n")
            f.write("\n" + "="*60 + "\n")
        
        # 保存为 pickle 以便后续处理
        metrics_dict = {
            'best_recall': best_recall,
            'best_top1_acc': best_top1_acc,
            'best_top10_acc': best_top10_acc,
            'best_epoch': best_recall_idx,
            'fold_id': int('${FOLD_ID}')
        }
        
        with open(output_dir / 'metrics.pkl', 'wb') as f:
            pickle.dump(metrics_dict, f)
        
        print(f"✓ 指标提取完成: {results_file}")
        print(f"  Top 1% Recall: {best_recall:.4f}")
    else:
        print(f"⚠ 未找到模型文件: {model_file}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ 提取指标时出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT

    EXTRACT_STATUS=$?
    
    if [ $EXTRACT_STATUS -eq 0 ]; then
        echo "✓ 指标提取成功"
    else
        echo "✗ 指标提取失败"
    fi
fi

echo ""
echo "任务 ${SLURM_ARRAY_TASK_ID} 完成"
echo "=================================================="

exit $TRAIN_STATUS