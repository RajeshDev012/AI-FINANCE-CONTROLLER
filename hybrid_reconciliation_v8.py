import re
import os
import pandas as pd
import numpy as np
import xgboost as xgb

from rapidfuzz.fuzz import token_set_ratio


# ============================================================
# CONFIG
# ============================================================

INVOICES_FILE = "invoices.csv"
TRANSACTIONS_FILE = "bank_transactions.csv"

MODEL_FILE = "xgboost_v7_model.json"
RESULT_FILE = "hybrid_v8_results.csv"

XGB_THRESHOLD = 0.50
MIN_MARGIN = 0.08
MAX_AMOUNT_DIFFERENCE_PERCENT = 5.0


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
        "BANKTRANSFER"
    ]:
        text = text.replace(term, " ")

    text = re.sub(
        r"[^A-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\n"
            f"Run xgboost_reconciliation.py first."
        )

    model = xgb.XGBClassifier()

    model.load_model(MODEL_FILE)

    print(
        f"Loaded XGBoost model: {MODEL_FILE}"
    )

    return model


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    invoices = pd.read_csv(
        INVOICES_FILE
    )

    transactions = pd.read_csv(
        TRANSACTIONS_FILE
    )

    invoices.columns = (
        invoices.columns.str.strip()
    )

    transactions.columns = (
        transactions.columns.str.strip()
    )

    print("=" * 70)
    print("AI FINANCE CONTROLLER V8")
    print("HYBRID XGBOOST + SAFETY CONTROLLER")
    print("=" * 70)

    print(
        f"Invoices:     {len(invoices)}"
    )

    print(
        f"Transactions: {len(transactions)}"
    )

    return invoices, transactions


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    invoices,
    transactions
):

    inv = invoices.copy()
    txs = transactions.copy()

    inv["invoice_date"] = pd.to_datetime(
        inv["invoice_date"],
        errors="coerce"
    )

    txs["transaction_date"] = pd.to_datetime(
        txs["transaction_date"],
        errors="coerce"
    )

    inv["vendor_normalized"] = (
        inv["vendor_name"]
        .fillna("")
        .apply(normalize_text)
    )

    inv["invoice_number_normalized"] = (
        inv["invoice_number"]
        .fillna("")
        .apply(normalize_text)
    )

    txs["description_normalized"] = (
        txs["description"]
        .fillna("")
        .apply(normalize_text)
    )

    rows = []

    print(
        "\nGenerating candidates..."
    )

    for _, tx in txs.iterrows():

        tx_amount = float(
            tx["amount"]
        )

        tx_date = tx[
            "transaction_date"
        ]

        tx_description = tx[
            "description_normalized"
        ]

        candidates = []

        for _, invoice in inv.iterrows():

            invoice_amount = float(
                invoice["invoice_amount"]
            )

            amount_difference = abs(
                tx_amount -
                invoice_amount
            )

            if invoice_amount != 0:

                amount_difference_percent = (
                    amount_difference /
                    invoice_amount
                ) * 100

            else:

                amount_difference_percent = 100

            vendor_similarity = token_set_ratio(
                tx_description,
                invoice[
                    "vendor_normalized"
                ]
            )

            if (
                pd.notna(tx_date)
                and
                pd.notna(
                    invoice["invoice_date"]
                )
            ):

                date_difference_days = abs(
                    (
                        tx_date -
                        invoice["invoice_date"]
                    ).days
                )

            else:

                date_difference_days = 999

            # ----------------------------------------------
            # Amount score
            # ----------------------------------------------

            if amount_difference <= 1:

                amount_score = 100

            else:

                amount_score = max(
                    0,
                    100 -
                    amount_difference_percent
                )

            # ----------------------------------------------
            # Date score
            # ----------------------------------------------

            date_score = max(
                0,
                100 -
                date_difference_days * 5
            )

            # ----------------------------------------------
            # Reference match
            # ----------------------------------------------

            invoice_number = invoice[
                "invoice_number_normalized"
            ]

            reference_match = int(
                invoice_number != ""
                and
                invoice_number in tx_description
            )

            # ----------------------------------------------
            # Explainable rule confidence
            # ----------------------------------------------

            if reference_match:

                confidence = (
                    amount_score * 0.45
                    +
                    vendor_similarity * 0.30
                    +
                    date_score * 0.10
                    +
                    100 * 0.15
                )

            else:

                total_weight = (
                    0.45 + 0.30 + 0.10
                )

                confidence = (
                    amount_score *
                    (0.45 / total_weight)
                    +
                    vendor_similarity *
                    (0.30 / total_weight)
                    +
                    date_score *
                    (0.10 / total_weight)
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
                    confidence
            })

        candidate_df = pd.DataFrame(
            candidates
        )

        # Rank according to explainable confidence
        candidate_df = candidate_df.sort_values(
            "confidence",
            ascending=False
        ).reset_index(
            drop=True
        )

        if len(candidate_df) > 1:

            best_score = candidate_df.loc[
                0,
                "confidence"
            ]

            second_score = candidate_df.loc[
                1,
                "confidence"
            ]

        else:

            best_score = candidate_df.loc[
                0,
                "confidence"
            ]

            second_score = 0

        margin = (
            best_score -
            second_score
        )

        candidate_df[
            "second_best_score"
        ] = second_score

        candidate_df[
            "confidence_margin"
        ] = margin

        rows.extend(
            candidate_df.to_dict(
                "records"
            )
        )

    result = pd.DataFrame(
        rows
    )

    print(
        f"Candidate pairs: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# XGBOOST PREDICTION
# ============================================================

def score_candidates(
    model,
    candidates
):

    X = candidates[
        FEATURE_COLUMNS
    ].fillna(0)

    candidates = candidates.copy()

    candidates[
        "xgb_probability"
    ] = model.predict_proba(X)[:, 1]

    return candidates


# ============================================================
# SELECT BEST CANDIDATE
# ============================================================

def select_best_candidate(
    candidates
):

    ranked = candidates.sort_values(
        [
            "transaction_id",
            "xgb_probability"
        ],
        ascending=[
            True,
            False
        ]
    ).copy()

    ranked["rank"] = (
        ranked
        .groupby("transaction_id")
        .cumcount()
        + 1
    )

    best = ranked[
        ranked["rank"] == 1
    ].copy()

    second = ranked[
        ranked["rank"] == 2
    ][
        [
            "transaction_id",
            "xgb_probability"
        ]
    ].rename(
        columns={
            "xgb_probability":
                "second_xgb_probability"
        }
    )

    best = best.merge(
        second,
        on="transaction_id",
        how="left"
    )

    best[
        "second_xgb_probability"
    ] = best[
        "second_xgb_probability"
    ].fillna(0)

    best[
        "model_margin"
    ] = (
        best["xgb_probability"]
        -
        best["second_xgb_probability"]
    )

    return best


# ============================================================
# SAFETY CONTROLLER
# ============================================================

def apply_safety_controller(
    best_candidates
):

    result = best_candidates.copy()

    result[
        "decision"
    ] = "REVIEW"

    result[
        "decision_reason"
    ] = "Requires analyst review"

    # Track invoices already assigned
    used_invoices = set()

    # Process transactions one by one
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

        invoice_id = row[
            "invoice_id"
        ]

        # ====================================================
        # SAFETY CHECK 1
        # ====================================================

        if probability < XGB_THRESHOLD:

            result.at[
                idx,
                "decision"
            ] = "REVIEW"

            result.at[
                idx,
                "decision_reason"
            ] = (
                "XGBoost confidence below threshold"
            )

            continue

        # ====================================================
        # SAFETY CHECK 2
        # ====================================================

        if margin < MIN_MARGIN:

            result.at[
                idx,
                "decision"
            ] = "REVIEW"

            result.at[
                idx,
                "decision_reason"
            ] = (
                "Ambiguous candidate ranking"
            )

            continue

        # ====================================================
        # SAFETY CHECK 3
        # ====================================================

        if (
            amount_difference
            >
            MAX_AMOUNT_DIFFERENCE_PERCENT
        ):

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

            continue

        # ====================================================
        # SAFETY CHECK 4
        # ====================================================

        if invoice_id in used_invoices:

            result.at[
                idx,
                "decision"
            ] = "REVIEW"

            result.at[
                idx,
                "decision_reason"
            ] = (
                "Invoice already assigned "
                "to another transaction"
            )

            continue

        # ====================================================
        # SAFE MATCH
        # ====================================================

        result.at[
            idx,
            "decision"
        ] = "MATCH"

        result.at[
            idx,
            "decision_reason"
        ] = (
            "XGBoost match passed "
            "financial safety checks"
        )

        used_invoices.add(
            invoice_id
        )

    return result


# ============================================================
# EXPLAIN DECISION
# ============================================================

def explain_decision(row):

    reasons = []

    if row[
        "amount_difference_percent"
    ] <= 1:

        reasons.append(
            "Exact/near-exact amount"
        )

    elif row[
        "amount_difference_percent"
    ] <= 5:

        reasons.append(
            "Amount within tolerance"
        )

    if row[
        "vendor_similarity"
    ] >= 80:

        reasons.append(
            "Strong vendor similarity"
        )

    elif row[
        "vendor_similarity"
    ] >= 60:

        reasons.append(
            "Moderate vendor similarity"
        )

    if row[
        "date_difference_days"
    ] <= 3:

        reasons.append(
            "Close transaction date"
        )

    if row[
        "reference_match"
    ] == 1:

        reasons.append(
            "Invoice reference found"
        )

    if not reasons:

        reasons.append(
            "Insufficient supporting evidence"
        )

    return " + ".join(reasons)


# ============================================================
# AUDIT OUTPUT
# ============================================================

def create_audit_output(
    results
):

    results = results.copy()

    results[
        "explanation"
    ] = results.apply(
        explain_decision,
        axis=1
    )

    output_columns = [

        "transaction_id",

        "invoice_id",

        "xgb_probability",

        "model_margin",

        "amount_difference_percent",

        "vendor_similarity",

        "date_difference_days",

        "reference_match",

        "decision",

        "decision_reason",

        "explanation"
    ]

    output = results[
        output_columns
    ]

    output.to_csv(
        RESULT_FILE,
        index=False
    )

    return output


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results
):

    total = len(results)

    matches = int(
        (
            results["decision"]
            == "MATCH"
        ).sum()
    )

    reviews = int(
        (
            results["decision"]
            == "REVIEW"
        ).sum()
    )

    print("\n")
    print("=" * 70)
    print("AI FINANCE CONTROLLER V8")
    print("HYBRID RECONCILIATION SUMMARY")
    print("=" * 70)

    print(
        f"Transactions processed: {total}"
    )

    print(
        f"MATCH:                 {matches}"
    )

    print(
        f"REVIEW:                {reviews}"
    )

    print(
        f"UNMATCHED:             0"
    )

    print("-" * 70)

    if total > 0:

        print(
            f"Automatic match rate: "
            f"{matches / total:.2%}"
        )

        print(
            f"Review rate: "
            f"{reviews / total:.2%}"
        )

    print("-" * 70)

    print(
        f"Average XGBoost probability: "
        f"{results['xgb_probability'].mean():.2%}"
    )

    match_rows = results[
        results["decision"] == "MATCH"
    ]

    if len(match_rows) > 0:

        print(
            f"Average MATCH probability: "
            f"{match_rows['xgb_probability'].mean():.2%}"
        )

    print("=" * 70)

    print(
        f"\nResults saved to: "
        f"{RESULT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    invoices, transactions = (
        load_data()
    )

    model = load_model()

    candidates = generate_candidates(
        invoices,
        transactions
    )

    candidates = score_candidates(
        model,
        candidates
    )

    best = select_best_candidate(
        candidates
    )

    results = apply_safety_controller(
        best
    )

    results = create_audit_output(
        results
    )

    print_summary(
        results
    )


if __name__ == "__main__":
    main()