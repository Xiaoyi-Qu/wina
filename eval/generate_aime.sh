#!/bin/bash

# DIR='pwd'

CHECKPOINT_DIR=$1
TOKENIZER_DIR=$1
seed=$2
DATA_NAME=$3
OUTPUT_FOLDER_NAME=$4
MODEL_TYPE=$5

DATA="data/${DATA_NAME}.jsonl"
#######################################
BSZ=30
TOTAL=30
GPUS=4
OUT_SEQ_LEN=8192
top_p=0.95
temperature=0.6
#######################################

SPARSITY_LEVELS=(0.0 0.1 0.2 0.3 0.5 0.7 0.9)

for sparsity in "${SPARSITY_LEVELS[@]}"; do
  local_seed=$seed
  for (( gpu=0; gpu<GPUS; gpu++ )); do
    python inference.py \
      --load "${CHECKPOINT_DIR}" \
      --tokenizer-model "${TOKENIZER_DIR}" \
      --max-output-len "${OUT_SEQ_LEN}" \
      --batch-size "${BSZ}" \
      --temperature "${temperature}" \
      --topp "${top_p}" \
      --tensor-parallel-size 1 \
      --seed "${local_seed}" \
      --bf16 \
      --model-type "${MODEL_TYPE}" \
      --output-folder "${OUTPUT_FOLDER_NAME}_sparsity_${sparsity}" \
      --datapath "${DATA}" \
      --sparsity "${sparsity}" \
      --device-id "${gpu}" &

    local_seed=$(( local_seed + 1 ))
  done
  wait
done

wait
echo "All GPUs finished."

