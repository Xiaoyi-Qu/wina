Under this folder, we will evaluate the performance of sparse model on mathematical reasoning tasks. In addition, we conduct empirical analysis to understand how the model's sparsity level/activation values it correlated to tokenwise entropy. 

## Evaluation
```bash
bash run_aime.sh
```

The results are saved into `outputs` folder. We evaluate the sparse model under different sparsity levels using AIME24 dataset. The evaluation metrics includes @Avg64 accuracy, tokenwise entropy, mean entropy, and 

## Empirical Analysis PartI
We first visualize how the sparsity level affects the distribution of tokenwise entropy. One can run the following command to obtain the results.

```bash
python visualization.py data1.jsonl data2.jsonl
```

The results are saved into `visualization` folder. 

## Empirical Analysis PartII
We further conduct analysis how the activation values affects the entropy values. Please run the following command.

```bash
python entropy_vs_activation.py
```

The results are saved into ???. 