# Sparsity Analysis for Reasoning Models

## Step 1: Obtain Sparsity

Compute the sparsity for each weight matrix by following the instructions in the `README.md` file under the `wina/` folder.

### Candidate Models

- `Qwen/Qwen3-VL-4B-Thinking`
- `mistralai/Ministral-3-3B-Reasoning-2512`
- `Qwen/Qwen3-VL-8B-Thinking`
- `mistralai/Ministral-3-8B-Reasoning-2512`

> **Note:** The current `wina` project is based on `transformers==4.44.1`. The code must be updated to support the new candidate models, which require a newer version of `transformers`.

## Step 2: Evaluation

Evaluate the performance of sparsified models on math reasoning tasks. We use **AIME24** and **AIME25** as benchmarks.

The existing bash script targets `Qwen/Qwen2.5-Math-7B-Instruct`. Add bash scripts in the same format for the four candidate models listed above, then run:

```bash
bash run_aime_qwen2.5-7B.sh
```

Results are saved to the `outputs/` folder with the following structure:

```
outputs/
├── aime24_sparsity_0.0/
│   ├── accuracy_results.json
│   ├── seed121.jsonl
│   └── seed122.jsonl
└── aime25_sparsity_0.0/
```

Evaluation metrics include:

- **Avg@32 accuracy**
- **Token-wise entropy**
- **Mean entropy**
- **Generated token length**

## Step 3: Empirical Analysis I — Entropy Distribution

Visualize how sparsity level affects the distribution of token-wise entropy. Tokens are color-coded by entropy level (low/high).

```bash
bash visualization.sh
```

Results are saved to the `visualization/` folder with the following structure:

```
visualization/
├── aime24_sparsity_0.0/
│   ├── seed121.jsonl/
│   └── seed122.jsonl/
│       ├── output_0.html
│       └── output_1.html
└── aime25_sparsity_0.0/
```

## Step 4: Empirical Analysis II — Activation vs Entropy

Analyze the correlation between activation values and entropy values. We select `mistralai/Ministral-3-3B-Reasoning-2512` model for this empirical analysis. 

```bash
python entropy_vs_activation.py
```

Results are saved to the `activation_entropy/` folder.