# COMP 472 Project: Image and Sentiment Classification


| Task | Classes | Features | Train | Test |
|---|---|---|---|---|
| Image (CORe50 objects) | 50 | 500 (sparse visual-word histograms) | 20,000 | 5,000 |
| Sentiment | 3 (negative / neutral / positive) | 768 (BERT embeddings) | 12,000 | 5,000 |

Kaggle score = (image accuracy + sentiment accuracy) / 2.

## Results (validation accuracy, shared 80/20 split)

| Model | Image | Sentiment |
|---|---|---|
| k-NN (cosine) | 0.534 (0.572 with square-root features) | 0.584 |
| Decision tree | 0.113 | 0.483 |
| Random forest | 0.170 (0.262 on Fisher features) | 0.598 (0.671 on Fisher features) |
| Neural network (PyTorch MLP) | 0.452 (0.442 on square-root features) | **0.683** |
| Local-mean k-NN | 0.591 | 0.588 |
| **Final: 0.7 local-mean k-NN + 0.3 MLP, square-root features** | **0.626** | — |

Final submission: image = blend above, sentiment = MLP. Validation estimate ≈ 0.654; the
submitted models are retrained on all labelled rows.

## Project structure
```
src/common/        shared code: data loading, the one validation split, preprocessing,
                   metrics, result logging, submission builder
src/models/        models: knn.py, decision_tree.py, random_forest.py, kmeans.py
experiments/       
scripts/           setup: check_setup.py, make_splits.py
splits/            saved train/validation indices, every model uses the same split
results/           
outputs/           boost_histories/: MLP training curves the final submission needs. Figures,
                   summaries and MLP tables are written here when experiments run.
```

## Setup
Tested with Python 3.13 on CPU.
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Download the 5 CSVs from the Kaggle **Data** tab into `data/`, then:
```bash
python scripts/check_setup.py     # verifies files and shapes
python scripts/make_splits.py     # creates the shared train/val split (already in splits/)
```

## Experiments
```bash
python experiments/knn_experiments.py          # k-NN: preprocessing, distance, k sweep
python experiments/tree_experiments.py         # Decision Tree + Random Forest
python experiments/clustering_experiments.py   # k-Means clustering
python experiments/tree_experiments.py --fisher  # Random Forest on Fisher (LDA) features
python experiments/mlp_experiments.py          # Neural network: image configs, sentiment CV
python experiments/score_boost.py --task image # model comparison for the final model
python experiments/score_boost.py --blend      # blend-weight table (how 0.7 was chosen)
python experiments/make_final_submission.py    # final Kaggle submission
```
Every run is logged to `results/results.csv`; plots go to `outputs/figures/`.
