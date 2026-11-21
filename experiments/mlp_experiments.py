"""Neural network (feed-forward MLP, PyTorch) for both tasks.

    python experiments/mlp_experiments.py --task image       # 256_64 vs 1024_reg, 3 seeds each
    python experiments/mlp_experiments.py --task sentiment   # 3-fold cross-validation

Results go to outputs/mlp/. The final submission reuses MLP, fit and predict from this file.
"""
import argparse
import copy
import json
from pathlib import Path
import platform
import random

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SEEDS = [42, 123, 2026]   # results are averaged over 3 starts, so one lucky start doesn't decide
BATCH_SIZE = 128          # common default: stable gradients, fast on CPU
MAX_EPOCHS = 50
PATIENCE = 8              # early stopping: stop after 8 epochs without validation improvement
SPECS = {"image": (20000, 500, 50), "sentiment": (12000, 768, 3)}
IMAGE_CONFIGS = [
    dict(name="256_64", hidden_sizes=[256, 64], dropout=0.0,
         weight_decay=0.0, learning_rate=0.001, num_classes=50),
    # wider layer + dropout (also on the inputs) + weight decay: the 256_64 net memorized the data
    dict(name="1024_reg", hidden_sizes=[1024], dropout=0.5, input_dropout=0.2,
         weight_decay=0.001, learning_rate=0.001, num_classes=50),
]
# Sentiment: a small net (64 units) with strong weight decay. The BERT features are almost
# linearly separable (a linear model already gets ~0.67), so a bigger net only overfits.
SENTIMENT_CONFIG = dict(
    name="sentiment", hidden_sizes=[64], dropout=0.3,
    weight_decay=0.01, learning_rate=0.0003, num_classes=3,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        layers = []
        # optional dropout on the input features: hides random features each batch
        if config.get("input_dropout", 0) > 0:
            layers.append(nn.Dropout(config["input_dropout"]))
        for size in config["hidden_sizes"]:
            layers.extend([nn.Linear(input_size, size), nn.ReLU()])
            if config["dropout"] > 0:
                layers.append(nn.Dropout(config["dropout"]))
            input_size = size
        layers.append(nn.Linear(input_size, config["num_classes"]))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_data(data_dir, task):
    df = pd.read_csv(data_dir / f"{task}_train.csv", dtype={"id": str})
    columns = [c for c in df.columns if c.startswith("feature_")]
    return (df[columns].to_numpy(dtype=np.float32), df["label"].to_numpy(dtype=np.int64),
            df["id"].to_numpy())


def fit_scaler(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def scale(X, mean, std):
    result = np.array((X - mean) / std, dtype=np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite standardized features.")
    return result


def loader(X, y, seed, shuffle=True, batch_size=BATCH_SIZE):
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def train_epoch(model, batches, optimizer):
    model.train()
    total = 0.0
    for X, y in batches:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(X), y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(batches.dataset)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        probabilities = np.concatenate([
            torch.softmax(model(torch.from_numpy(X[i:i + 512])), dim=1).numpy()
            for i in range(0, len(X), 512)
        ])
    if (not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise RuntimeError("Invalid predicted probabilities.")
    return probabilities


def evaluate(model, X, y):
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(X), 512):
            labels = torch.from_numpy(y[start:start + 512])
            outputs = model(torch.from_numpy(X[start:start + 512]))
            loss_sum += nn.functional.cross_entropy(outputs, labels).item() * len(labels)
            correct += int((outputs.argmax(1) == labels).sum())
    return loss_sum / len(y), correct / len(y)


def fit(X, y, config, seed, history_path, validation=None, epochs=None,
        report_train=False):
    """Early stop on supplied validation, or train for a fixed epoch count."""
    set_seed(seed)
    model = MLP(X.shape[1], config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    batches = loader(X, y, seed)
    # keep the weights from the best validation epoch and restore them at the end, so later
    # overfitting epochs don't count
    best_accuracy, best_epoch, stale, best_state = -1.0, 0, 0, None
    history = []
    limit = MAX_EPOCHS if validation is not None else epochs
    if limit is None or limit < 1:
        raise ValueError("A positive fixed epoch count is required.")
    for epoch in range(1, limit + 1):
        loss = train_epoch(model, batches, optimizer)
        row = {"epoch": epoch, "training_loss_with_dropout": loss}
        message = f"Epoch {epoch:02d}/{limit} | Training loss: {loss:.4f}"
        if report_train:
            row["train_loss"], row["train_accuracy"] = evaluate(model, X, y)
            message += f" | Train: {row['train_accuracy']:.2%}"
        if validation is not None:
            val_loss, accuracy = evaluate(model, *validation)
            row.update(val_loss=val_loss, val_accuracy=accuracy)
            message += f" | Validation: {accuracy:.2%} | Val loss: {val_loss:.4f}"
            if accuracy > best_accuracy:
                best_accuracy, best_epoch, stale = accuracy, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        history.append(row)
        print(message, flush=True)
        if validation is not None and stale >= PATIENCE:
            print(f"Early stopping: no improvement for {PATIENCE} epochs.")
            break
    pd.DataFrame(history).to_csv(history_path, index=False)
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = limit
    return model, best_epoch, best_accuracy


def save_analysis(directory, name, ids, y, probabilities):
    predictions = probabilities.argmax(axis=1)
    classes = list(range(probabilities.shape[1]))
    report = classification_report(y, predictions, labels=classes,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(directory / f"{name}_report.csv")
    pd.DataFrame(confusion_matrix(y, predictions, labels=classes),
                 index=classes, columns=classes).to_csv(
                     directory / f"{name}_confusion_matrix.csv", index_label="true_label")
    frame = pd.DataFrame(dict(id=ids, true_label=y, predicted_label=predictions))
    for c in classes:
        frame[f"prob_{c}"] = probabilities[:, c]
    frame.to_csv(directory / f"{name}_predictions.csv", index=False)
    return dict(accuracy=float(np.mean(predictions == y)),
                macro_f1=report["macro avg"]["f1-score"],
                neutral_recall=report["1"]["recall"] if len(classes) == 3 else None)


def image_validation(args, results):
    X, y, ids = load_data(args.data_dir, "image")
    train, val = train_test_split(np.arange(len(y)), test_size=0.2,
                                 stratify=y, random_state=SEED)
    mean, std = fit_scaler(X[train])
    X_train, X_val = scale(X[train], mean, std), scale(X[val], mean, std)
    pd.DataFrame({"id": ids, "partition": np.where(np.isin(np.arange(len(y)), val),
                 "validation", "training")}).to_csv(results / "image_split.csv", index=False)
    validation_probs, rows = [], []
    for config in IMAGE_CONFIGS:
        for seed in SEEDS:
            name = f"image_{config['name']}_seed_{seed}"
            print(f"\n--- {name} ---")
            model, epoch, _ = fit(X_train, y[train], config, seed,
                results / f"{name}_history.csv", validation=(X_val, y[val]), report_train=True)
            probabilities = predict(model, X_val)
            validation_probs.append(probabilities)
            rows.append(dict(architecture=config["name"], seed=seed, best_epoch=epoch,
                             accuracy=float(np.mean(probabilities.argmax(1) == y[val]))))
    pd.DataFrame(rows).to_csv(results / "image_individual_results.csv", index=False)
    comparisons = []
    for name, probabilities in [
        ("image_old_3", np.mean(np.stack(validation_probs[:3]), axis=0)),
        ("image_new_3", np.mean(np.stack(validation_probs[3:]), axis=0)),
        ("image_combined_6", np.mean(np.stack(validation_probs), axis=0)),
    ]:
        metrics = save_analysis(results, name, ids[val], y[val], probabilities)
        comparisons.append(dict(model=name, **metrics))
    summary = pd.DataFrame(comparisons).drop(columns="neutral_recall")
    summary.to_csv(results / "image_summary.csv", index=False)
    print("\n--- IMAGE HOLDOUT VALIDATION ---\n" + summary.to_string(index=False))


def sentiment_validation(args, results):
    X, y, ids = load_data(args.data_dir, "sentiment")
    # Nested cross-validation: the epoch count is chosen on an inner split and accuracy is
    # measured on the outer fold the model never saw, for two weight-decay strengths.
    weights = [0.001, 0.01]
    oof = {wd: np.full((len(y), 3), np.nan, dtype=np.float32) for wd in weights}
    folds = np.zeros(len(y), dtype=int)
    rows = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (train, outer) in enumerate(splitter.split(X, y), start=1):
        folds[outer] = fold
        inner_train, inner_val = train_test_split(train, test_size=0.2,
                                                 stratify=y[train], random_state=SEED)
        mean_i, std_i = fit_scaler(X[inner_train])
        mean_o, std_o = fit_scaler(X[train])
        for wd in weights:
            config = dict(SENTIMENT_CONFIG, weight_decay=wd)
            name = f"sentiment_fold_{fold}_wd_{wd:g}"
            print(f"\n--- {name}: inner epoch selection ---")
            _, epoch, _ = fit(scale(X[inner_train], mean_i, std_i), y[inner_train],
                config, SEED, results / f"{name}_selection.csv",
                validation=(scale(X[inner_val], mean_i, std_i), y[inner_val]))
            print(f"Refitting on outer training partition for {epoch} epochs.")
            model, _, _ = fit(scale(X[train], mean_o, std_o), y[train], config, SEED,
                             results / f"{name}_refit.csv", epochs=epoch)
            probabilities = predict(model, scale(X[outer], mean_o, std_o))
            oof[wd][outer] = probabilities
            metrics = save_analysis(results, name, ids[outer], y[outer], probabilities)
            rows.append(dict(fold=fold, weight_decay=wd, selected_epochs=epoch, **metrics))
            print(f"Outer validation accuracy: {metrics['accuracy']:.3%}")
            pd.DataFrame(rows).to_csv(results / "sentiment_fold_results.csv", index=False)
    summary = []
    frame = pd.DataFrame(rows)
    for wd in weights:
        if not np.isfinite(oof[wd]).all():
            raise RuntimeError("Missing out-of-fold predictions.")
        metrics = save_analysis(results, f"sentiment_oof_wd_{wd:g}", ids, y, oof[wd])
        scores = frame.loc[frame.weight_decay == wd, "accuracy"]
        summary.append(dict(weight_decay=wd, mean_accuracy=scores.mean(),
                            std_accuracy=scores.std(ddof=1), **metrics))
    pd.DataFrame(dict(id=ids, fold=folds)).to_csv(results / "sentiment_folds.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(results / "sentiment_cv_summary.csv", index=False)
    print("\n--- SENTIMENT CROSS-VALIDATION ---\n" + summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["image", "sentiment", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    results = ROOT / "outputs" / "mlp"
    results.mkdir(parents=True, exist_ok=True)
    environment = dict(python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, torch=str(torch.__version__), sklearn=sklearn.__version__,
        device="cpu", task=args.task, seed=SEED)
    (results / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    if args.task in ("image", "both"):
        image_validation(args, results)
    if args.task in ("sentiment", "both"):
        sentiment_validation(args, results)


if __name__ == "__main__":
    main()
