"""
ML upgrade path for AI-FINANCE-CONTROLLER.

This file does three things:
  1. Fixes the confidence-scoring bug (reference weight not renormalized
     when no reference is present) that was causing false negatives.
  2. Logs every REVIEW/MATCH decision + the analyst's correction to a
     growing labeled dataset (labels.csv).
  3. Once you have enough labeled rows, trains an XGBoost classifier on
     the same features your rule engine already computes, and lets you
     compare it against the rule engine.

Nothing here requires you to throw away your existing V5 engine — run
both side by side. Use the rule engine's output as your production
decision, log corrections, and once the model's precision/recall beats
the rules on held-out data, switch over.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    xgb = None  # only needed once you actually train


LABELS_PATH = "labels.csv"
MODEL_PATH = "reconciliation_model.json"

FEATURE_COLUMNS = [
    "amount_score",
    "vendor_similarity",
    "date_score",
    "reference_match",
    "confidence",
    "confidence_margin",
]


# =============================================================================
# 1. FIXED CONFIDENCE FUNCTION (drop-in replacement for your original)
# =============================================================================
def calculate_confidence(
    amt_score: float,
    vendor_score: float,
    date_score: float,
    ref_score: float,
) -> float:
    """
    Weighted confidence with renormalization when no reference is found.

    Bug this fixes: the original always applied WEIGHT_REF=0.15 even when
    ref_score=0 because no invoice number appeared in the bank description
    (the common case). That silently capped correct matches ~15 points
    below where they should land. Here, when there's no reference evidence,
    its weight is redistributed across amount/vendor/date instead of being
    wasted.
    """
    ref_present = ref_score > 0

    if ref_present:
        w_amt, w_vendor, w_date, w_ref = 0.45, 0.30, 0.10, 0.15
    else:
        # Redistribute the 0.15 proportionally to the other three
        # (0.45, 0.30, 0.10 -> scaled up to sum to 1.0)
        total = 0.45 + 0.30 + 0.10
        w_amt = 0.45 / total
        w_vendor = 0.30 / total
        w_date = 0.10 / total
        w_ref = 0.0

    score = (
        amt_score * w_amt
        + vendor_score * w_vendor
        + date_score * w_date
        + ref_score * w_ref
    )
    return round(score, 2)


# =============================================================================
# 2. LABEL LOGGING
# =============================================================================
def log_labeled_decision(
    transaction_id: str,
    invoice_id: Optional[str],
    features: Dict[str, Any],
    human_label: int,
    notes: str = "",
) -> None:
    """
    Append one human-confirmed outcome to labels.csv.

    Call this every time an analyst clears a REVIEW item (or spot-checks
    an automatic MATCH). human_label: 1 = this was a true match,
    0 = it was not a match.

    features should be the dict of per-candidate metrics your
    generate_candidates() already produces: amount_score, vendor_similarity,
    date_score (or date_difference_days), reference_match, confidence,
    confidence_margin.
    """
    row = {
        "transaction_id": transaction_id,
        "invoice_id": invoice_id,
        **{col: features.get(col) for col in FEATURE_COLUMNS},
        "human_label": human_label,
        "notes": notes,
    }

    df_row = pd.DataFrame([row])
    if os.path.exists(LABELS_PATH):
        df_row.to_csv(LABELS_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LABELS_PATH, mode="w", header=True, index=False)


# =============================================================================
# 3. TRAINING
# =============================================================================
def train_model(labels_path: str = LABELS_PATH, min_rows: int = 80) -> Optional["xgb.XGBClassifier"]:
    """
    Train an XGBoost classifier on logged, human-confirmed labels.

    min_rows: don't bother training until you have a reasonable amount of
    labeled data — with fewer than ~80 examples the model will just
    memorize noise. Keep using the rule engine until then.
    """
    if xgb is None:
        raise ImportError("pip install xgboost scikit-learn --break-system-packages")

    if not os.path.exists(labels_path):
        print(f"No labels file at {labels_path} yet. Log some decisions first.")
        return None

    df = pd.read_csv(labels_path)
    df["reference_match"] = df["reference_match"].astype(int)

    if len(df) < min_rows:
        print(f"Only {len(df)} labeled rows so far (need ~{min_rows}+). "
              f"Keep using the rule engine and logging outcomes.")
        return None

    X = df[FEATURE_COLUMNS]
    y = df["human_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n=== Held-out performance ===")
    print(classification_report(y_test, preds, digits=3))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, preds))

    model.save_model(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Feature importance — tells you which signal actually drives matches
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print("\nFeature importance:")
    print(importances.sort_values(ascending=False))

    return model


def predict_match_probability(model: "xgb.XGBClassifier", features: Dict[str, Any]) -> float:
    """Given the same feature dict used above, return P(this is a true match)."""
    row = pd.DataFrame([{col: features.get(col) for col in FEATURE_COLUMNS}])
    row["reference_match"] = row["reference_match"].astype(int)
    return float(model.predict_proba(row)[0][1])


if __name__ == "__main__":
    # Example flow once you have labels.csv populated:
    model = train_model()
    if model is not None:
        example_features = {
            "amount_score": 100.0,
            "vendor_similarity": 78.0,
            "date_score": 90.0,
            "reference_match": 0,
            "confidence": 80.5,
            "confidence_margin": 12.0,
        }
        prob = predict_match_probability(model, example_features)
        print(f"\nExample prediction: {prob:.2%} probability of true match")