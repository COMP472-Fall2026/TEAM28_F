"""Build a Kaggle submission. Matches predictions to IDs by MERGE, never by row order."""
from datetime import datetime

import numpy as np
import pandas as pd

from .config import SUBMISSIONS_DIR, TASKS
from .data import load_sample_submission, load_test


def build_submission(image_preds, sentiment_preds, name: str = "submission") -> str:
    """image_preds / sentiment_preds must be in the same row order as image_test.csv /
    sentiment_test.csv (which is what load_test() returns)."""
    img_ids, _ = load_test("image")
    txt_ids, _ = load_test("sentiment")
    image_preds = np.asarray(image_preds)
    sentiment_preds = np.asarray(sentiment_preds)
    assert len(image_preds) == len(img_ids), f"{len(image_preds)} image preds vs {len(img_ids)} ids"
    assert len(sentiment_preds) == len(txt_ids), f"{len(sentiment_preds)} sentiment preds vs {len(txt_ids)} ids"

    preds = pd.concat([
        pd.DataFrame({"id": img_ids, "label": image_preds}),
        pd.DataFrame({"id": txt_ids, "label": sentiment_preds}),
    ], ignore_index=True)

    sample = load_sample_submission()
    # match predictions to ids by merging, never by row position: the sample file mixes image
    # and text rows in its own order
    sub = sample[["id"]].merge(preds, on="id", how="left", validate="one_to_one")

    # Same checks as the official example, plus a couple more.
    expected = TASKS["image"]["n_test"] + TASKS["sentiment"]["n_test"]
    assert len(sub) == len(sample), "row count differs from sample_submission"
    if len(sample) != expected:
        print(f"[warn] sample_submission has {len(sample)} rows (expected {expected}).")
    assert sub["label"].notna().all(), "missing predictions for some ids"
    assert sub["id"].is_unique, "duplicate ids"
    sub["label"] = sub["label"].astype(int)
    img = sub["id"].str.startswith(TASKS["image"]["test_id_prefix"])
    txt = sub["id"].str.startswith(TASKS["sentiment"]["test_id_prefix"])
    assert (img | txt).all(), "unexpected id prefix"
    assert sub.loc[img, "label"].between(0, 49).all(), "image label out of range"
    assert sub.loc[txt, "label"].between(0, 2).all(), "sentiment label out of range"

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = SUBMISSIONS_DIR / f"{name}_{stamp}.csv"
    sub.to_csv(out, index=False)
    print(f"[submission] wrote {out} ({len(sub)} rows). Upload via 'Submit Predictions' on Kaggle.")
    return str(out)
