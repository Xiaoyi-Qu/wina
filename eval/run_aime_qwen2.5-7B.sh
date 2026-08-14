#! /bin/bash
# - https://huggingface.co/datasets/HuggingFaceH4/MATH-500
# - https://huggingface.co/datasets/math-ai/minervamath
# - https://huggingface.co/datasets/Hothan/OlympiadBench

# MODEL_NAME=$"nvidia/AceReason-Nemotron-7B"
MODEL_NAME=$"Qwen/Qwen2.5-Math-7B-Instruct" # Please replace with your own model name
OUTPUT_FOLDER_NAME=$"outputs_qwen2.5-7B"    # Please replace with your own 
BACKEND=$"hf"

if [ "$MODEL_NAME" == "Qwen/Qwen2.5-Math-7B-Instruct" ]; then
    seed_list_aime24=(121 131 141 151 161 171 181 191)
    seed_list_aime25=(111 222 333 444 555 666 777 888)
    MODEL_TYPE="qwen"
fi

SPARSITY_LEVELS=(0.0 0.1 0.2 0.3 0.5 0.7 0.9)

# AIME 24
for seed in ${seed_list_aime24[@]}; do
    bash generate_aime.sh ${MODEL_NAME} ${seed} aime24 "${OUTPUT_FOLDER_NAME}/aime24" ${MODEL_TYPE}
done
for sparsity in "${SPARSITY_LEVELS[@]}"; do
  python evaluate_aime.py \
    --modelfolder "${OUTPUT_FOLDER_NAME}/aime24_sparsity_${sparsity}" \
    --test_data data/aime24.jsonl
done

# AIME 25
for seed in ${seed_list_aime25[@]}; do
    bash generate_aime.sh ${MODEL_NAME} ${seed} aime25 "${OUTPUT_FOLDER_NAME}/aime25" ${MODEL_TYPE}
done
for sparsity in "${SPARSITY_LEVELS[@]}"; do
  python evaluate_aime.py \
    --modelfolder "${OUTPUT_FOLDER_NAME}/aime25_sparsity_${sparsity}" \
    --test_data data/aime25.jsonl
done

# # AMC 23
# for seed in ${seed_list_amc23[@]}; do
#     bash generate_aime.sh ${MODEL_NAME} ${seed} amc23 "${OUTPUT_FOLDER_NAME}/amc23" ${MODEL_TYPE}
# done
# python evaluate_aime.py --modelfolder "${OUTPUT_FOLDER_NAME}/amc23" --test_data data/amc23.jsonl


# # math500 (set GPUS=1 in generate_aime.sh file)
# bash generate_aime.sh ${MODEL_NAME} ${seed_math500} math500 "${OUTPUT_FOLDER_NAME}/math500" ${MODEL_TYPE}
# python evaluate_aime.py --modelfolder "${OUTPUT_FOLDER_NAME}/math500" --test_data data/math500.jsonl


# # Minervamath (set GPUS=1 in generate_aime.sh file)
# bash generate_aime.sh ${MODEL_NAME} ${seed_minervamath} minervamath "${OUTPUT_FOLDER_NAME}/minervamath" ${MODEL_TYPE}
# python evaluate_aime.py --modelfolder "${OUTPUT_FOLDER_NAME}/minervamath" --test_data data/minervamath.jsonl


# # olympiad (set GPUS=1 in generate_aime.sh file)
# bash generate_aime.sh ${MODEL_NAME} ${seed_olympiad} olympiad "${OUTPUT_FOLDER_NAME}/olympiad" ${MODEL_TYPE}
# python evaluate_aime.py --modelfolder "${OUTPUT_FOLDER_NAME}/olympiad" --test_data data/olympiad.jsonl