#! /bin/bash
# - https://huggingface.co/datasets/HuggingFaceH4/MATH-500
# - https://huggingface.co/datasets/math-ai/minervamath
# - https://huggingface.co/datasets/Hothan/OlympiadBench

# MODEL_NAME=$"nvidia/AceReason-Nemotron-7B"
MODEL_NAME=$"Qwen/Qwen2.5-Math-7B-Instruct" # $"../train/ckpt/distilled_student_w4g128"
OUTPUT_FOLDER_NAME=$"outputs"
BACKEND=$"hf"

if [ "$MODEL_NAME" == "Qwen/Qwen2.5-Math-7B-Instruct" ]; then
    seed_list_aime24=(121) # 131 141 151 161 171 181 191)
    seed_list_aime25=(111) # 222 333 444 555 666 777 888)
    # seed_list_amc23=(101)  # 201 301 401 501 601 701 801
    # seed_math500=(151)
    # seed_minervamath=(131)
    # seed_olympiad=(141)
    MODEL_TYPE="qwen"
elif [ "$MODEL_NAME" == "ckpt/distilled_student_w4g128" ]; then
    # seed_list_aime24=(121) # 131 141 151 161 171 181 191)
    # seed_list_aime25=(111) # 222 333 444 555 666 777 888)
    # seed_list_amc23=(101) # 201 301 401 501 601 701 801)
    seed_list_aime24=(121 131 141 151 161 171 181 191 201 211 221 231 241 251 261 271)
    seed_list_aime25=(111 222 333 444 555 666 777 888 999 1110 1221 1332 1443 1554 1665 1776)
    seed_list_amc23=(101 201 301 401 501 601 701 801 901 1001 1101 1201 1301 1401 1501 1601)
    seed_math500=(151)
    seed_minervamath=(131)
    seed_olympiad=(141)
    MODEL_TYPE="qwen"
elif [ "$MODEL_NAME" == "ckpt/distilled_student_w3g128" ]; then
    # seed_list_aime24=(121) # 131 141 151 161 171 181 191)
    # seed_list_aime25=(111) # 222 333 444 555 666 777 888)
    # seed_list_amc23=(101) # 201 301 401 501 601 701 801)
    seed_list_aime24=(121 131 141 151 161 171 181 191 201 211 221 231 241 251 261 271)
    seed_list_aime25=(111 222 333 444 555 666 777 888 999 1110 1221 1332 1443 1554 1665 1776)
    seed_list_amc23=(101 201 301 401 501 601 701 801 901 1001 1101 1201 1301 1401 1501 1601)
    seed_math500=(151)
    seed_minervamath=(131)
    seed_olympiad=(141)
    MODEL_TYPE="qwen"
elif [ "$MODEL_NAME" == "nvidia/AceReason-Nemotron-14B" ]; then
    seed_list_aime24=(111 222 333 444 555 666 777 888)
    seed_list_aime25=(111 222 333 444 555 666 777 888)
    MODEL_TYPE="r1"
elif [ "$MODEL_NAME" == "nvidia/AceReason-Nemotron-1.1-7B" ]; then
    seed_list_aime24=(100 200 300 400 500 600 700 800)
    seed_list_aime25=(100 200 300 400 500 600 700 800)
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