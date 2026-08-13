#! /bin/bash

for dataset in aime24 aime25; do
    for dir in outputs/${dataset}_sparsity_*/; do
        for file in "$dir"*.jsonl; do
            [ -f "$file" ] || continue
            python visualization.py -i "$file" -o output.html
        done
    done
done