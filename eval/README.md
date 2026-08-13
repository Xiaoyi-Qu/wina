# Sparse Model Evaluation on Mathematical Reasoning

This folder contains experiments evaluating sparse models on mathematical reasoning tasks. We also conduct empirical analysis to understand how sparsity level and activation values correlate with token-wise entropy.

## Evaluation

We evaluate the sparsified model on the `AIME24` and `AIME25` math reasoning benchmarks.

```bash
bash run_aime.sh  # Specify the number of GPUs in generate_aime.sh
```

Results are saved to the `outputs/` folder. Evaluation metrics include:
- **Avg@32 accuracy**
- **Token-wise entropy**
- **Mean entropy**
- **Generated token length**

## Empirical Analysis Part I

We visualize how sparsity level affects the distribution of token-wise entropy. Run the following command to generate the results:

```bash
bash visualization.sh
```

Results are saved to the `visualization/` folder. Tokens are color-coded by entropy level (low / high).

<!-- ## Empirical Analysis PartII
We further conduct analysis on how the activation values affects the entropy values. Please run the following command.

```bash
python entropy_vs_activation.py
```

The results are saved into `???` folder.  -->