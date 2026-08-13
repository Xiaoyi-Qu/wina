Under this folder, we will evaluate the performance of sparse model on mathematical reasoning tasks. In addition, we conduct empirical analysis to understand how the model's sparsity level/activation values is correlated to tokenwise entropy. 

## Evaluation
We evaluate the sparsified model on `AIME24` and `AIME25` math reasoning tasks.
```bash
bash run_aime.sh # Need to specify the number of GPUs in the generate_aime.sh folder
```

The results are saved into `outputs` folder. The evaluation metrics includes @Avg32 accuracy, tokenwise entropy, mean entropy, and generated token length. 

## Empirical Analysis PartI
We first visualize how the sparsity level affects the distribution of tokenwise entropy. One can run the following command to obtain the results.

```bash
bash visualization.sh 
```

The results are saved into `visualization` folder. The text is labeled using different color to distinguish between different entropy level. 

<!-- ## Empirical Analysis PartII
We further conduct analysis on how the activation values affects the entropy values. Please run the following command.

```bash
python entropy_vs_activation.py
```

The results are saved into `???` folder.  -->