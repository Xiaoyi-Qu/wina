The first step is to obtain the sparsity for each weight matrices. Please follow instructions provided in the `README.md` file under `wina` folder.

Our new candidate models include:
- `Qwen/Qwen3-VL-4B-Thinking`
- `mistralai/Ministral-3-3B-Reasoning-2512`
- `Qwen/Qwen3-VL-8B-Thinking`
- `mistralai/Ministral-3-8B-Reasoning-2512`

The current `wina` project is based on the package `transformers 4.44.1`. One needs to edit the code such that new candidate models (based on new version of transformers) are supported.

## Evaluation

The next step is to evaluate the performance of sparsified model on math reasoning tasks. We select `AIME24` and `AIME25` as our math reasoning benchmarks. Note that the current bash script targets for `Qwen/Qwen2.5-Math-7B-Instruct` model. Please add bash scripts of same format for the four candidate models. After adding bash scripts for new models, run the following command to obtain the evaluation results.

```bash
bash run_aime.sh
```

Results are saved to the `outputs/` folder. Evaluation metrics include:
- **Avg@32 accuracy**
- **Token-wise entropy**
- **Mean entropy**
- **Generated token length**

The structure of the `outputs/` folder is organized as follows:
    outputs/
    ├── aime24_sparsity_0.0/
    │   ├── accuracy_results.json
    │   ├── seed121.jsonl
    │   └── seed122.jsonl
    └── aime25_sparsity_0.0/

## Empirical Analysis Part I

In the first empirical analysis, we visualize how sparsity level affects the distribution of token-wise entropy. Run the following command to generate the results.

```bash
bash visualization.sh
```

Results are saved in the `visualization/` folder. Tokens are color-coded by entropy level (low / high).

## Empirical Analysis Part II
In the second empirical analysis, we conduct analysis on how the activation values are correlated with the entropy values. Please run the following command.

```bash
python entropy_vs_activation.py
```

The results are saved into under `neuron_count_entropy` folder. 