import pandas as pd
import numpy as np
import time


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_FILE = "reconciliation_results.csv"
GROUND_TRUTH_FILE = "ground_truth.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    results = pd.read_csv(RESULTS_FILE)
    ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

    required_result_columns = [
        "transaction_id",
        "predicted_invoice_id",
        "confidence",
        "decision",
    ]

    required_ground_truth_columns = [
    "transaction_id",
    "actual_invoice_id",
    "actual_status",
    ]

    for column in required_result_columns:
        if column not in results.columns:
            raise ValueError(
                f"Missing column in reconciliation_results.csv: {column}"
            )

    for column in required_ground_truth_columns:
        if column not in ground_truth.columns:
            raise ValueError(
                f"Missing column in ground_truth.csv: {column}"
            )

    return results, ground_truth


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

def prepare_ground_truth(ground_truth):
    """
    Prepare the clean transaction-level ground truth.

    Each bank transaction has exactly one ground-truth record.
    """

    ground_truth = ground_truth.copy()

    ground_truth["transaction_id"] = (
        ground_truth["transaction_id"]
        .astype(str)
        .str.strip()
    )

    ground_truth["actual_invoice_id"] = (
        ground_truth["actual_invoice_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    ground_truth["actual_status"] = (
        ground_truth["actual_status"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return ground_truth


# ============================================================
# CREATE EVALUATION LABELS
# ============================================================

def create_evaluation_table(results, ground_truth):
    """
    Merge model predictions with clean transaction-level
    ground truth.

    Each CREDIT transaction should have exactly one
    ground-truth record.
    """

    evaluation = results.copy()

    # Normalize prediction IDs
    evaluation["transaction_id"] = (
        evaluation["transaction_id"]
        .astype(str)
        .str.strip()
    )

    evaluation["predicted_invoice_id"] = (
        evaluation["predicted_invoice_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Prepare ground truth
    truth = ground_truth[
        [
            "transaction_id",
            "actual_invoice_id",
            "actual_status",
        ]
    ].copy()

    truth["transaction_id"] = (
        truth["transaction_id"]
        .astype(str)
        .str.strip()
    )

    truth["actual_invoice_id"] = (
        truth["actual_invoice_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    truth["actual_status"] = (
        truth["actual_status"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Merge prediction with ground truth
    evaluation = evaluation.merge(
        truth,
        on="transaction_id",
        how="left"
    )

    # Handle missing values
    evaluation["actual_invoice_id"] = (
        evaluation["actual_invoice_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    evaluation["actual_status"] = (
        evaluation["actual_status"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # IMPORTANT: return the evaluation dataframe
    return evaluation


# ============================================================
# CALCULATE MATCH METRICS
# ============================================================

def calculate_match_metrics(evaluation):

    # --------------------------------------------------------
    # A TRUE POSITIVE means:
    #
    # Model predicted MATCH
    # AND
    # predicted invoice equals actual invoice
    # --------------------------------------------------------

    true_positive = (
        (evaluation["decision"] == "MATCH")
        &
        (evaluation["predicted_invoice_id"]
         == evaluation["actual_invoice_id"])
        &
        (evaluation["actual_status"] == "MATCH")
    )

    # --------------------------------------------------------
    # FALSE POSITIVE
    #
    # Model predicted MATCH but the match was incorrect.
    # --------------------------------------------------------

    false_positive = (
        (evaluation["decision"] == "MATCH")
        &
        (
            (evaluation["predicted_invoice_id"]
             != evaluation["actual_invoice_id"])
            |
            (evaluation["actual_status"] != "MATCH")
        )
    )

    # --------------------------------------------------------
    # FALSE NEGATIVE
    #
    # Ground truth says there is a valid match,
    # but model did not correctly MATCH it.
    # --------------------------------------------------------

    false_negative = (
        (evaluation["actual_status"] == "MATCH")
        &
        ~true_positive
    )

    tp = int(true_positive.sum())
    fp = int(false_positive.sum())
    fn = int(false_negative.sum())

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if precision + recall > 0:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# DECISION DISTRIBUTION
# ============================================================

def calculate_decision_metrics(evaluation):

    total = len(evaluation)

    match_count = (
        evaluation["decision"] == "MATCH"
    ).sum()

    review_count = (
        evaluation["decision"] == "REVIEW"
    ).sum()

    unmatched_count = (
        evaluation["decision"] == "UNMATCHED"
    ).sum()

    if total > 0:
        match_rate = match_count / total
        exception_rate = (
            (review_count + unmatched_count)
            / total
        )
    else:
        match_rate = 0
        exception_rate = 0

    return {
        "total_records": total,
        "match_count": int(match_count),
        "review_count": int(review_count),
        "unmatched_count": int(unmatched_count),
        "match_rate": match_rate,
        "exception_rate": exception_rate,
    }


# ============================================================
# ANALYZE FAILURE TYPES
# ============================================================

def analyze_failures(evaluation):

    print()
    print("=" * 60)
    print("GROUND-TRUTH STATUS DISTRIBUTION")
    print("=" * 60)

    status_counts = (
        evaluation["actual_status"]
        .value_counts()
    )

    for status, count in status_counts.items():
        print(f"{status:<25} {count}")

    print()
    print("=" * 60)
    print("FALSE POSITIVE CASES")
    print("=" * 60)

    fp_cases = evaluation[
        (evaluation["decision"] == "MATCH")
        &
        (
            (evaluation["predicted_invoice_id"]
             != evaluation["actual_invoice_id"])
            |
            (evaluation["actual_status"] != "MATCH")
        )
    ]

    if len(fp_cases) == 0:
        print("No false positives.")
    else:
        print(
            fp_cases[
                [
                    "transaction_id",
                    "predicted_invoice_id",
                    "actual_invoice_id",
                    "confidence",
                    "actual_status",
                ]
            ].head(20).to_string(index=False)
        )

    print()
    print("=" * 60)
    print("FALSE NEGATIVE CASES")
    print("=" * 60)

    fn_cases = evaluation[
        (evaluation["actual_status"] == "MATCH")
        &
        ~(
            (evaluation["decision"] == "MATCH")
            &
            (
                evaluation["predicted_invoice_id"]
                == evaluation["actual_invoice_id"]
            )
        )
    ]

    if len(fn_cases) == 0:
        print("No false negatives.")
    else:
        print(
            fn_cases[
                [
                    "transaction_id",
                    "predicted_invoice_id",
                    "actual_invoice_id",
                    "confidence",
                    "decision",
                ]
            ].head(20).to_string(index=False)
        )


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

def analyze_confidence(evaluation):

    print()
    print("=" * 60)
    print("CONFIDENCE ANALYSIS")
    print("=" * 60)

    matched = evaluation[
        evaluation["decision"] == "MATCH"
    ]

    if len(matched) == 0:
        print("No MATCH records available.")
        return

    print(
        f"Average confidence of MATCH decisions: "
        f"{matched['confidence'].mean():.2f}"
    )

    print(
        f"Minimum confidence of MATCH decisions: "
        f"{matched['confidence'].min():.2f}"
    )

    print(
        f"Maximum confidence of MATCH decisions: "
        f"{matched['confidence'].max():.2f}"
    )


# ============================================================
# SAVE EVALUATION DATA
# ============================================================

def save_evaluation(evaluation):

    evaluation.to_csv(
        "reconciliation_evaluation.csv",
        index=False
    )

    print()
    print(
        "Detailed evaluation saved to:"
    )

    print(
        "reconciliation_evaluation.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print()
    print("=" * 60)
    print("AI FINANCE CONTROLLER")
    print("RECONCILIATION EVALUATION")
    print("=" * 60)

    # Load
    results, ground_truth = load_data()

    # Prepare
    ground_truth = prepare_ground_truth(
        ground_truth
    )

    # Create evaluation table
    evaluation = create_evaluation_table(
        results,
        ground_truth
    )

    # Calculate metrics
    match_metrics = calculate_match_metrics(
        evaluation
    )

    decision_metrics = calculate_decision_metrics(
        evaluation
    )

    # --------------------------------------------------------
    # PRINT MAIN RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(
        f"Total records:       "
        f"{decision_metrics['total_records']}"
    )

    print(
        f"Automatic MATCH:     "
        f"{decision_metrics['match_count']}"
    )

    print(
        f"REVIEW:              "
        f"{decision_metrics['review_count']}"
    )

    print(
        f"UNMATCHED:           "
        f"{decision_metrics['unmatched_count']}"
    )

    print()

    print(
        f"Match rate:           "
        f"{decision_metrics['match_rate'] * 100:.2f}%"
    )

    print(
        f"Exception rate:      "
        f"{decision_metrics['exception_rate'] * 100:.2f}%"
    )

    print()
    print("=" * 60)
    print("MODEL QUALITY")
    print("=" * 60)

    print(
        f"True positives:      "
        f"{match_metrics['true_positive']}"
    )

    print(
        f"False positives:     "
        f"{match_metrics['false_positive']}"
    )

    print(
        f"False negatives:     "
        f"{match_metrics['false_negative']}"
    )

    print()

    print(
        f"Precision:            "
        f"{match_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:               "
        f"{match_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score:             "
        f"{match_metrics['f1'] * 100:.2f}%"
    )

    # Additional analysis
    analyze_failures(evaluation)

    analyze_confidence(evaluation)

    # Save
    save_evaluation(evaluation)

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print(
        f"Evaluation time: {elapsed:.4f} seconds"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()