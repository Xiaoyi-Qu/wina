#! /bin/bash

INPUT_FOLDER_NAME=$"outputs_qwen2.5-7B"

for dataset in aime24 aime25; do
    for dir in ${INPUT_FOLDER_NAME}/${dataset}_sparsity_*/; do
        for file in "$dir"*.jsonl; do
            [ -f "$file" ] || continue
            python visualization.py -i "$file" -o output.html
        done
    done
done