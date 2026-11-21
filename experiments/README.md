One script per study. Run each from the repo root; every run is logged to `results/results.csv`.

| Script | What it does |
|---|---|
| `knn_experiments.py` | k-NN: raw vs standardized features, euclidean vs cosine, k sweep, weighted voting, error analysis → `outputs/knn_summary.md` |
| `tree_experiments.py` | Decision tree depth/split/leaf/criterion sweep and random forest (number of trees, hard vs soft voting) |
| `clustering_experiments.py` | k-Means on both tasks: clusters vs true labels, k sweep → `outputs/clustering_summary.md` |
| `mlp_experiments.py` | Neural network (PyTorch): image configs 256_64 vs 1024_reg, sentiment cross-validation |
| `score_boost.py` | Model comparison for the final model: local-mean k-NN, linear model, MLP variants; `--blend` prints the blend-weight table |
| `make_final_submission.py` | Re-checks the chosen models on validation, refits on all labelled rows, writes the Kaggle CSV |
| `_utils.py` | Shared helpers (path setup, markdown tables, plotting backend) |
