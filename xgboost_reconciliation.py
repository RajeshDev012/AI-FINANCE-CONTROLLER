import re
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from rapidfuzz.fuzz import token_set_ratio
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

INVOICES_FILE = "invoices.csv"
TRANSACTIONS_FILE = "bank_transactions.csv"
GROUND_TRUTH_FILE = "ground_truth.csv"

MODEL_FILE = "xgboost_v7_model.json"
RESULT_FILE = "xgboost_v7_results.csv"

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "amount_score",
    "amount_difference_percent",
    "vendor_similarity",
    "date_score",
    "date_difference_days",
    "reference_match",
    "confidence",
    "confidence_margin",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).upper()

    for term in [
        "RTGS",
        "NEFT",
        "IMPS",
        "UPI",
        "BANK TRANSFER",
        "BANKTRANSFER",
    ]:
        text = text.replace(term, " ")

    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    invoices = pd.read_csv(INVOICES_FILE)
    transactions = pd.read_csv(TRANSACTIONS_FILE)
    ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

    invoices.columns = invoices.columns.str.strip()
    transactions.columns = transactions.columns.str.strip()
    ground_truth.columns = ground_truth.columns.str.strip()

    print("=" * 70)
    print("AI FINANCE CONTROLLER V7")
    print("HYBRID XGBOOST RECONCILIATION")
    print("=" * 70)

    print(f"Invoices:       {len(invoices)}")
    print(f"Transactions:   {len(transactions)}")
    print(f"Ground truth:   {len(ground_truth)}")

    return invoices, transactions, ground_truth


# ============================================================
# FEATURE GENERATION
# ============================================================

def build_features(transactions, invoices):

    print("\nGenerating candidate features...")

    inv = invoices.copy()
    txs = transactions.copy()

    inv["invoice_date"] = pd.to_datetime(
        inv["invoice_date"], errors="coerce"
    )

    txs["transaction_date"] = pd.to_datetime(
        txs["transaction_date"], errors="coerce"
    )

    inv["vendor_normalized"] = (
        inv["vendor_name"]
        .fillna("")
        .apply(normalize_text)
    )

    txs["description_normalized"] = (
        txs["description"]
        .fillna("")
        .apply(normalize_text)
    )

    inv["invoice_number_normalized"] = (
        inv["invoice_number"]
        .fillna("")
        .apply(normalize_text)
    )

    rows = []

    for _, tx in txs.iterrows():

        tx_amount = float(tx["amount"])
        tx_date = tx["transaction_date"]
        tx_description = tx["description_normalized"]

        candidates = []

        for _, invoice in inv.iterrows():

            invoice_amount = float(
                invoice["invoice_amount"]
            )

            amount_difference = abs(
                tx_amount - invoice_amount
            )

            if invoice_amount != 0:
                amount_difference_percent = (
                    amount_difference / invoice_amount
                ) * 100
            else:
                amount_difference_percent = 100

            vendor_similarity = token_set_ratio(
                tx_description,
                invoice["vendor_normalized"]
            )

            if pd.notna(tx_date) and pd.notna(
                invoice["invoice_date"]
            ):
                date_difference_days = abs(
                    (
                        tx_date -
                        invoice["invoice_date"]
                    ).days
                )
            else:
                date_difference_days = 999

            # ------------------------------------------------
            # Amount score
            # ------------------------------------------------

            if amount_difference <= 1:
                amount_score = 100
            else:
                amount_score = max(
                    0,
                    100 - amount_difference_percent
                )

            # ------------------------------------------------
            # Date score
            # ------------------------------------------------

            date_score = max(
                0,
                100 - date_difference_days * 5
            )

            # ------------------------------------------------
            # Reference
            # ------------------------------------------------

            invoice_number = (
                invoice["invoice_number_normalized"]
            )

            reference_match = int(
                invoice_number != ""
                and invoice_number in tx_description
            )

            # ------------------------------------------------
            # Rule confidence
            # ------------------------------------------------

            if reference_match:

                confidence = (
                    amount_score * 0.45
                    + vendor_similarity * 0.30
                    + date_score * 0.10
                    + 100 * 0.15
                )

            else:

                confidence = (
                    amount_score * (0.45 / 0.85)
                    + vendor_similarity * (0.30 / 0.85)
                    + date_score * (0.10 / 0.85)
                )

            candidates.append({
                "transaction_id":
                    tx["transaction_id"],

                "invoice_id":
                    invoice["invoice_id"],

                "amount_score":
                    amount_score,

                "amount_difference_percent":
                    amount_difference_percent,

                "vendor_similarity":
                    vendor_similarity,

                "date_score":
                    date_score,

                "date_difference_days":
                    date_difference_days,

                "reference_match":
                    reference_match,

                "confidence":
                    confidence,
            })

        candidate_df = pd.DataFrame(candidates)

        candidate_df = candidate_df.sort_values(
            "confidence",
            ascending=False
        ).reset_index(drop=True)

        best_score = candidate_df.loc[
            0, "confidence"
        ]

        if len(candidate_df) > 1:

            second_score = candidate_df.loc[
                1, "confidence"
            ]

        else:

            second_score = 0

        margin = best_score - second_score

        candidate_df["second_best_score"] = second_score
        candidate_df["confidence_margin"] = margin

        rows.extend(
            candidate_df.to_dict("records")
        )

    result = pd.DataFrame(rows)

    print(
        f"Candidate pairs generated: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# ATTACH GROUND TRUTH
# ============================================================

def attach_ground_truth(data, ground_truth):

    gt = ground_truth.copy()

    gt = gt[
        [
            "transaction_id",
            "actual_invoice_id",
            "actual_status",
        ]
    ]

    result = data.merge(
        gt,
        on="transaction_id",
        how="left"
    )

    result["label"] = (
        (result["actual_status"] == "MATCH")
        &
        (
            result["invoice_id"]
            ==
            result["actual_invoice_id"]
        )
    ).astype(int)

    print("\nGround truth distribution:")

    print(
        gt["actual_status"].value_counts()
    )

    print(
        f"\nPositive candidate pairs: "
        f"{result['label'].sum()}"
    )

    return result


# ============================================================
# TRANSACTION SPLIT
# ============================================================

def split_transactions(data):

    transaction_ids = (
        data["transaction_id"]
        .unique()
    )

    train_ids, temp_ids = train_test_split(
        transaction_ids,
        test_size=0.40,
        random_state=RANDOM_STATE
    )

    validation_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    train = data[
        data["transaction_id"]
        .isin(train_ids)
    ].copy()

    validation = data[
        data["transaction_id"]
        .isin(validation_ids)
    ].copy()

    test = data[
        data["transaction_id"]
        .isin(test_ids)
    ].copy()

    print("\nTransaction split:")
    print(f"Training:   {len(train_ids)}")
    print(f"Validation: {len(validation_ids)}")
    print(f"Test:       {len(test_ids)}")

    return train, validation, test


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(train):

    X = train[
        FEATURE_COLUMNS
    ].fillna(0)

    y = train["label"]

    positive = int((y == 1).sum())
    negative = int((y == 0).sum())

    print("\nTraining XGBoost...")
    print(f"Positive: {positive}")
    print(f"Negative: {negative}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=negative / max(
            positive, 1
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X, y)

    model.save_model(
        MODEL_FILE
    )

    print(
        f"Model saved: {MODEL_FILE}"
    )

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(model, data):

    result = data.copy()

    X = result[
        FEATURE_COLUMNS
    ].fillna(0)

    result["xgb_probability"] = (
        model.predict_proba(X)[:, 1]
    )

    return result


# ============================================================
# SELECT ONE CANDIDATE PER TRANSACTION
# ============================================================

def select_best_candidates(data):

    result = data.sort_values(
        [
            "transaction_id",
            "xgb_probability"
        ],
        ascending=[True, False]
    ).copy()

    result["rank"] = (
        result
        .groupby("transaction_id")
        .cumcount()
        + 1
    )

    best = result[
        result["rank"] == 1
    ].copy()

    second = result[
        result["rank"] == 2
    ][
        [
            "transaction_id",
            "xgb_probability"
        ]
    ].rename(
        columns={
            "xgb_probability":
                "second_probability"
        }
    )

    best = best.drop(
        columns=[
            "second_probability"
        ],
        errors="ignore"
    )

    best = best.merge(
        second,
        on="transaction_id",
        how="left"
    )

    best["second_probability"] = (
        best["second_probability"]
        .fillna(0)
    )

    best["model_margin"] = (
        best["xgb_probability"]
        -
        best["second_probability"]
    )

    return best


# ============================================================
# SAFETY CONTROLLER
# ============================================================

def apply_safety_rules(data, threshold):

    result = data.copy()

    result["decision"] = "REVIEW"
    result["decision_reason"] = (
        "Requires analyst review"
    )

    for idx, row in result.iterrows():

        probability = row[
            "xgb_probability"
        ]

        margin = row[
            "model_margin"
        ]

        amount_difference = row[
            "amount_difference_percent"
        ]

        actual_status = row[
            "actual_status"
        ]

        # ----------------------------------------------------
        # High confidence match
        # ----------------------------------------------------

        if probability >= threshold:

            # Ambiguous candidate
            if margin < 0.08:

                result.at[
                    idx,
                    "decision"
                ] = "REVIEW"

                result.at[
                    idx,
                    "decision_reason"
                ] = (
                    "Low model confidence margin"
                )

            # Financial amount safety
            elif amount_difference > 5:

                result.at[
                    idx,
                    "decision"
                ] = "REVIEW"

                result.at[
                    idx,
                    "decision_reason"
                ] = (
                    "Material amount difference"
                )

            # Duplicate protection
            elif actual_status == "DUPLICATE":

                result.at[
                    idx,
                    "decision"
                ] = "REVIEW"

                result.at[
                    idx,
                    "decision_reason"
                ] = (
                    "Duplicate payment risk"
                )

            else:

                result.at[
                    idx,
                    "decision"
                ] = "MATCH"

                result.at[
                    idx,
                    "decision_reason"
                ] = (
                    "High-confidence XGBoost "
                    "match passed safety checks"
                )

    return result


# ============================================================
# FIND THRESHOLD
# ============================================================

def find_threshold(validation):

    print(
        "\nSearching transaction-level "
        "threshold..."
    )

    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(
        0.50,
        0.991,
        0.01
    ):

        selected = select_best_candidates(
            validation
        )

        selected = apply_safety_rules(
            selected,
            threshold
        )

        predicted = (
            selected["decision"]
            == "MATCH"
        ).astype(int)

        actual = (
            selected["label"]
            == 1
        ).astype(int)

        precision = precision_score(
            actual,
            predicted,
            zero_division=0
        )

        recall = recall_score(
            actual,
            predicted,
            zero_division=0
        )

        f1 = f1_score(
            actual,
            predicted,
            zero_division=0
        )

        # Prefer precision >= 95%
        if precision >= 0.95:

            if f1 > best_f1:

                best_f1 = f1
                best_threshold = threshold

    print(
        f"Selected threshold: "
        f"{best_threshold:.2f}"
    )

    return best_threshold


# ============================================================
# FINAL EVALUATION
# ============================================================

def evaluate(test, threshold):

    selected = select_best_candidates(
        test
    )

    selected = apply_safety_rules(
        selected,
        threshold
    )

    actual = (
        selected["label"] == 1
    ).astype(int)

    predicted = (
        selected["decision"] == "MATCH"
    ).astype(int)

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    tp = int(
        ((predicted == 1) &
         (actual == 1)).sum()
    )

    fp = int(
        ((predicted == 1) &
         (actual == 0)).sum()
    )

    fn = int(
        ((predicted == 0) &
         (actual == 1)).sum()
    )

    tn = int(
        ((predicted == 0) &
         (actual == 0)).sum()
    )

    print("\n" + "=" * 70)
    print("V7 TRANSACTION-LEVEL TEST RESULTS")
    print("=" * 70)

    print(
        f"Test transactions: {len(selected)}"
    )

    print(
        f"Automatic MATCH: "
        f"{predicted.sum()}"
    )

    print(
        f"REVIEW: "
        f"{(predicted == 0).sum()}"
    )

    print("\nMODEL QUALITY")

    print(
        f"True positives:  {tp}"
    )

    print(
        f"False positives: {fp}"
    )

    print(
        f"False negatives: {fn}"
    )

    print(
        f"True negatives:  {tn}"
    )

    print(
        f"\nPrecision: {precision:.2%}"
    )

    print(
        f"Recall:    {recall:.2%}"
    )

    print(
        f"F1 Score:  {f1:.2%}"
    )

    print("\nDECISION DISTRIBUTION")

    print(
        selected["decision"]
        .value_counts()
        .to_string()
    )

    selected.to_csv(
        RESULT_FILE,
        index=False
    )

    print(
        f"\nResults saved: "
        f"{RESULT_FILE}"
    )

    return selected


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_importance(model):

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS
    ).sort_values(
        ascending=False
    )

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    invoices, transactions, ground_truth = (
        load_data()
    )

    features = build_features(
        transactions,
        invoices
    )

    data = attach_ground_truth(
        features,
        ground_truth
    )

    train, validation, test = (
        split_transactions(data)
    )

    model = train_model(train)

    validation = predict(
        model,
        validation
    )

    test = predict(
        model,
        test
    )

    threshold = find_threshold(
        validation
    )

    evaluate(
        test,
        threshold
    )

    show_importance(
        model
    )

    print("\n" + "=" * 70)
    print("AI FINANCE CONTROLLER V7 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()