Please follow the README file on the main page to obtain sparsity allocation the target model.

## Evaluation
We test the sparsed model on `AIME24` math reasoning task using Average64 accuracy as the metric. Along with that, we will record information includeing tokenwise entropy, mean entropy, and length of generated tokens.  
```bash
bash run_aime.sh
```

The test results will be saved into `outputs/aime24_sparsity_{SPARSITY_LEVEL}` folder.

## Empircal Analysis PartI
We study how the sparsity level affects the resulting model token entropy distribution. To achieve this, we run the following command.
```bash
python visualization.py
```

The results will be saved into which folder. 

## Empircal Analysis PartII
We further study how the resulting token entropy is correlated to the activation value proceeded by MLP layer.