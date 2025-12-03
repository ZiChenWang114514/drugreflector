理解你的问题。HPC的QOS政策限制了job array的使用。我会为你重写一个不使用job array的版本，改用循环批量提交单个任务的方式。

以下是新的解决方案，包含两个文件：

## 文件1: 单个任务脚本 (`run_single_hyperparam.sh`)

```bash
#!/bin/bash
#SBATCH -J drugreflector-hp
#SBATCH -p gpu_4l
#SBATCH -N 1 
#SBATCH -o ../err_out/hyperparam_1fold_%j.out
#SBATCH -e ../err_out/hyperparam_1fold_%j.err
#SBATCH --no-requeue
#SBATCH -A tangc_g1
#SBATCH --qos=tangcg4c
#SBATCH --gres=gpu:1
#SBATCH --overcommit
#SBATCH --mincpus=28

# ============================================================================
# DrugReflector 单个超参数配置训练脚本
# 由主脚本调用，每次训练一个配置
# ============================================================================

source /appsnew/source/Miniconda3-py38_4.10.3.sh
conda activate env_chemprop_4l

# ============================================================================
# 从命令行参数接收超参数配置
# ============================================================================

TASK_ID=$1
CURRENT_LR=$2
CURRENT_SCHEDULER=$3
CURRENT_WD=$4
CURRENT_DROPOUT=$5

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

# 固定其他超参数
FOCAL_GAMMA=2.0
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
OUTPUT_DIR="${BASE_OUTPUT_DIR}/config_${CONFIG_ID}_task${TASK_ID}"

# ============================================================================
# 打印当前配置
# ============================================================================
echo "=================================================="
echo "单折超参数优化任务"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "任务 ID: ${TASK_ID}"
echo "训练 Fold: ${FOLD_ID}"
echo "时间戳: ${TIMESTAMP}"
echo ""
echo "当前配置:"
echo "  学习率: ${CURRENT_LR}"
echo "  调度器: ${CURRENT_SCHEDULER}"
echo "  权重衰减: ${CURRENT_WD}"
echo "  Dropout: ${CURRENT_DROPOUT}"
echo "  Focal Gamma: ${FOCAL_GAMMA}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "=================================================="
echo ""

# ============================================================================
# 创建输出目录并保存配置
# ============================================================================

mkdir -p "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/config.txt" << EOF
================================================
超参数配置 - 任务 ${TASK_ID}
================================================
训练参数:
  数据文件: ${DATA_FILE}
  训练 Fold: ${FOLD_ID}
  训练轮次: ${EPOCHS}
  批次大小: ${BATCH_SIZE}

优化超参数:
  学习率: ${CURRENT_LR}
  学习率调度器: ${CURRENT_SCHEDULER}
  学习率衰减率: ${LR_DECAY_RATE}
  学习率衰减轮次: ${LR_DECAY_EPOCHS}
  最小学习率: ${CURRENT_MIN_LR}
  权重衰减: ${CURRENT_WD}
  Dropout: ${CURRENT_DROPOUT}
  Focal Gamma: ${FOCAL_GAMMA}

任务信息:
  SLURM Job ID: ${SLURM_JOB_ID}
  节点: ${SLURM_NODELIST}
  开始时间: $(date)
================================================
EOF

# ============================================================================
# 执行训练
# ============================================================================

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
    --focal-gamma ${FOCAL_GAMMA} \
    --plot-dir ${PLOT_DIR} \
    --device ${DEVICE} \
    --num-workers ${NUM_WORKERS} \
    --save-every 10"

if [ -n "$LR_DECAY_EPOCHS" ]; then
    CMD="${CMD} --lr-decay-epochs ${LR_DECAY_EPOCHS}"
fi

echo "执行命令:"
echo "${CMD}"
echo ""

START_TIME=$(date +%s)

eval ${CMD}

TRAIN_STATUS=$?

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED_TIME / 3600))
ELAPSED_MINS=$(((ELAPSED_TIME % 3600) / 60))
ELAPSED_SECS=$((ELAPSED_TIME % 60))

echo ""
echo "=================================================="
echo "训练完成 - 任务 ${TASK_ID}"
echo "  状态码: ${TRAIN_STATUS}"
echo "  用时: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m ${ELAPSED_SECS}s"
echo "=================================================="

cat >> "${OUTPUT_DIR}/config.txt" << EOF

结果:
  训练状态: $(if [ $TRAIN_STATUS -eq 0 ]; then echo "成功"; else echo "失败 (${TRAIN_STATUS})"; fi)
  完成时间: $(date)
  运行时长: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m ${ELAPSED_SECS}s
================================================
EOF

# ============================================================================
# 提取指标
# ============================================================================

if [ $TRAIN_STATUS -eq 0 ]; then
    echo "提取训练指标..."
    
    python -u ../drugreflector_train/extract_single_fold_metrics.py \
        --output-dir "${OUTPUT_DIR}" \
        --fold-id ${FOLD_ID}
    
    EXTRACT_STATUS=$?
    
    if [ $EXTRACT_STATUS -eq 0 ]; then
        echo "✓ 指标提取成功"
    else
        echo "✗ 指标提取失败"
    fi
fi

echo "任务 ${TASK_ID} 完成"

exit $TRAIN_STATUS
```

## 文件2: 主提交脚本 (`submit_all_hyperparams.sh`)

```bash
#!/bin/bash

# ============================================================================
# DrugReflector 超参数优化批量提交脚本
# 不使用 job array，而是循环提交单个任务
# ============================================================================

# 超参数搜索空间定义
LR_VALUES=(0.001 0.01 0.025 0.05)
LR_SCHEDULER_VALUES=("step" "exponential" "cosine")
WEIGHT_DECAY_VALUES=(1e-6 5e-5)
DROPOUT_VALUES=(0.3 0.5 0.7)

# 计算总配置数
N_LR=${#LR_VALUES[@]}
N_SCHEDULER=${#LR_SCHEDULER_VALUES[@]}
N_WD=${#WEIGHT_DECAY_VALUES[@]}
N_DROPOUT=${#DROPOUT_VALUES[@]}
TOTAL_CONFIGS=$((N_LR * N_SCHEDULER * N_WD * N_DROPOUT))

echo "=================================================="
echo "DrugReflector 超参数优化批量提交"
echo "=================================================="
echo "总配置数: ${TOTAL_CONFIGS}"
echo "学习率: ${LR_VALUES[@]}"
echo "调度器: ${LR_SCHEDULER_VALUES[@]}"
echo "权重衰减: ${WEIGHT_DECAY_VALUES[@]}"
echo "Dropout: ${DROPOUT_VALUES[@]}"
echo "=================================================="
echo ""

# 创建日志目录
mkdir -p ../err_out
mkdir -p ../models/hyperparam_search_1fold

# 记录所有提交的 job IDs
JOB_IDS_FILE="../models/hyperparam_search_1fold/submitted_jobs.txt"
echo "# 超参数优化任务提交记录 - $(date)" > ${JOB_IDS_FILE}
echo "# 总任务数: ${TOTAL_CONFIGS}" >> ${JOB_IDS_FILE}
echo "" >> ${JOB_IDS_FILE}

# 提交计数
SUBMITTED=0
FAILED=0

# 遍历所有超参数组合
TASK_ID=0
for lr in "${LR_VALUES[@]}"; do
    for scheduler in "${LR_SCHEDULER_VALUES[@]}"; do
        for wd in "${WEIGHT_DECAY_VALUES[@]}"; do
            for dropout in "${DROPOUT_VALUES[@]}"; do
                
                echo "[$((TASK_ID + 1))/${TOTAL_CONFIGS}] 提交配置: lr=${lr}, scheduler=${scheduler}, wd=${wd}, dropout=${dropout}"
                
                # 提交任务
                JOB_OUTPUT=$(sbatch run_single_hyperparam.sh ${TASK_ID} ${lr} ${scheduler} ${wd} ${dropout} 2>&1)
                
                # 检查提交是否成功
                if [ $? -eq 0 ]; then
                    JOB_ID=$(echo ${JOB_OUTPUT} | grep -oP 'Submitted batch job \K[0-9]+')
                    echo "  ✓ 提交成功: Job ID = ${JOB_ID}"
                    echo "Task_${TASK_ID}: Job_${JOB_ID} | lr=${lr} scheduler=${scheduler} wd=${wd} dropout=${dropout}" >> ${JOB_IDS_FILE}
                    SUBMITTED=$((SUBMITTED + 1))
                else
                    echo "  ✗ 提交失败: ${JOB_OUTPUT}"
                    echo "Task_${TASK_ID}: FAILED | lr=${lr} scheduler=${scheduler} wd=${wd} dropout=${dropout} | Error: ${JOB_OUTPUT}" >> ${JOB_IDS_FILE}
                    FAILED=$((FAILED + 1))
                fi
                
                TASK_ID=$((TASK_ID + 1))
                
                # 添加短暂延迟，避免过快提交
                sleep 0.5
                
            done
        done
    done
done

echo ""
echo "=================================================="
echo "提交完成"
echo "=================================================="
echo "成功提交: ${SUBMITTED}/${TOTAL_CONFIGS}"
echo "失败: ${FAILED}/${TOTAL_CONFIGS}"
echo "Job IDs 记录文件: ${JOB_IDS_FILE}"
echo "=================================================="
echo ""
echo "监控任务状态: squeue -u \$USER"
echo "查看特定任务: squeue -j <JOB_ID>"
echo "取消所有任务: scancel -u \$USER"
echo "=================================================="
```

## 文件3: 监控脚本 (`monitor_jobs.sh`)

```bash
#!/bin/bash

# ============================================================================
# 监控所有超参数优化任务的状态
# ============================================================================

JOB_IDS_FILE="../models/hyperparam_search_1fold/submitted_jobs.txt"

if [ ! -f "${JOB_IDS_FILE}" ]; then
    echo "错误: 找不到任务记录文件 ${JOB_IDS_FILE}"
    exit 1
fi

echo "=================================================="
echo "DrugReflector 超参数优化任务监控"
echo "=================================================="
echo ""

# 提取所有 Job IDs
JOB_IDS=$(grep "Job_" ${JOB_IDS_FILE} | grep -oP 'Job_\K[0-9]+')

if [ -z "${JOB_IDS}" ]; then
    echo "没有找到已提交的任务"
    exit 0
fi

# 统计各状态任务数
TOTAL=$(echo "${JOB_IDS}" | wc -l)
RUNNING=$(squeue -j $(echo ${JOB_IDS} | tr ' ' ',') -h -t RUNNING 2>/dev/null | wc -l)
PENDING=$(squeue -j $(echo ${JOB_IDS} | tr ' ' ',') -h -t PENDING 2>/dev/null | wc -l)
COMPLETED=$((TOTAL - RUNNING - PENDING))

echo "任务状态统计:"
echo "  总任务数: ${TOTAL}"
echo "  运行中: ${RUNNING}"
echo "  等待中: ${PENDING}"
echo "  已完成: ${COMPLETED}"
echo ""
echo "=================================================="
echo "当前运行/等待的任务:"
echo "=================================================="

squeue -j $(echo ${JOB_IDS} | tr ' ' ',') 2>/dev/null || echo "所有任务已完成"

echo ""
echo "=================================================="
echo "实时更新: watch -n 30 ./monitor_jobs.sh"
echo "取消所有任务: scancel -j \$(cat ${JOB_IDS_FILE} | grep 'Job_' | grep -oP 'Job_\K[0-9]+' | tr '\n' ',')"
echo "=================================================="
```

## 使用方法

1. **准备脚本**：

```bash
chmod +x run_single_hyperparam.sh
chmod +x submit_all_hyperparams.sh
chmod +x monitor_jobs.sh
```

1. **提交所有任务**：

```bash
./submit_all_hyperparams.sh
```

1. **监控任务状态**：

```bash
# 查看一次
./monitor_jobs.sh

# 持续监控（每30秒刷新）
watch -n 30 ./monitor_jobs.sh
```

1. **查看具体任务**：

```bash
# 查看所有任务
squeue -u $USER

# 查看输出日志
tail -f ../err_out/hyperparam_1fold_<JOB_ID>.out
```

## 优势

1. **避免 QOS 限制**：不使用 job array，每个任务独立提交
2. **灵活控制**：可以随时停止提交，或选择性提交某些配置
3. **易于追踪**：所有 Job IDs 记录在文件中，方便管理
4. **容错性好**：单个任务失败不影响其他任务
5. **可中断恢复**：可以修改脚本只提交未完成的配置

## 注意事项

- 73个任务会逐个提交，可能需要几分钟完成提交过程
- 确保HPC的任务队列限制允许同时提交这么多任务
- 如果仍有限制，可以分批提交（修改循环逻辑）

需要我提供分批提交的版本吗？