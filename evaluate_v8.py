import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


V8_FILE = "hybrid_v8_results.csv"
GROUND_TRUTH_FILE = "ground_truth.csv"


def main():

    v8 = pd.read_csv(V8_FILE)
    gt = pd.read_csv(GROUND_TRUTH_FILE)

    v8.columns = v8.columns.str.strip()
    gt.columns = gt.columns.str.strip()

    # ---------------------------------------------------------
    # Ground truth columns
    # ---------------------------------------------------------

    gt = gt[
        [
            "transaction_id",
            "actual_invoice_id",
            "actual_status"
        ]
    ].copy()

    # ---------------------------------------------------------
    # Merge V8 predictions with ground truth
    # ---------------------------------------------------------

    data = v8.merge(
        gt,
        on="transaction_id",
        how="inner"
    )

    print("=" * 65)
    print("AI FINANCE CONTROLLER V8")
    print("RECONCILIATION EVALUATION")
    print("=" * 65)

    print(
        f"V8 transactions processed: {len(v8)}"
    )

    print(
        f"Ground-truth transactions: {len(gt)}"
    )

    print(
        f"Transactions evaluated:    {len(data)}"
    )

    # ---------------------------------------------------------
    # Actual status
    # ---------------------------------------------------------

    data["actual_match"] = (
        data["actual_status"] == "MATCH"
    )

    # ---------------------------------------------------------
    # Predicted automatic match
    # ---------------------------------------------------------

    data["predicted_match"] = (
        data["decision"] == "MATCH"
    )

    # ---------------------------------------------------------
    # Correct match
    #
    # True positive:
    # actual status = MATCH
    # AND predicted MATCH
    # AND predicted invoice = actual invoice
    # ---------------------------------------------------------

    data["true_positive"] = (
        data["actual_match"]
        &
        data["predicted_match"]
        &
        (
            data["invoice_id"]
            ==
            data["actual_invoice_id"]
        )
    )

    # ---------------------------------------------------------
    # False positive
    #
    # System automatically matched something that was not
    # the correct ground-truth match.
    # ---------------------------------------------------------

    data["false_positive"] = (
        data["predicted_match"]
        &
        ~data["true_positive"]
    )

    # ---------------------------------------------------------
    # False negative
    #
    # Ground truth says MATCH but system didn't correctly
    # automatically match it.
    # ---------------------------------------------------------

    data["false_negative"] = (
        data["actual_match"]
        &
        ~data["true_positive"]
    )

    # ---------------------------------------------------------
    # True negative
    #
    # Actual transaction is not a normal MATCH and system
    # correctly did not automatically match it.
    # ---------------------------------------------------------

    data["true_negative"] = (
        ~data["actual_match"]
        &
        ~data["predicted_match"]
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    tp = int(data["true_positive"].sum())
    fp = int(data["false_positive"].sum())
    fn = int(data["false_negative"].sum())
    tn = int(data["true_negative"].sum())

    precision = precision_score(
        data["actual_match"],
        data["predicted_match"],
        zero_division=0
    )

    recall = recall_score(
        data["actual_match"],
        data["predicted_match"],
        zero_division=0
    )

    f1 = f1_score(
        data["actual_match"],
        data["predicted_match"],
        zero_division=0
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("V8 MODEL QUALITY")
    print("=" * 65)

    print(f"True positives:  {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True negatives:  {tn}")

    print()
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")

    # ---------------------------------------------------------
    # Decision distribution
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("V8 DECISION DISTRIBUTION")
    print("=" * 65)

    print(
        data["decision"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # Ground truth distribution
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("GROUND TRUTH DISTRIBUTION")
    print("=" * 65)

    print(
        data["actual_status"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # False positives
    # ---------------------------------------------------------

    false_positives = data[
        data["false_positive"]
    ]

    print("\n" + "=" * 65)
    print("FALSE POSITIVE CASES")
    print("=" * 65)

    if len(false_positives) == 0:

        print("None")

    else:

        print(
            false_positives[
                [
                    "transaction_id",
                    "invoice_id",
                    "actual_invoice_id",
                    "xgb_probability",
                    "actual_status",
                    "decision",
                    "decision_reason"
                ]
            ].to_string(index=False)
        )

    # ---------------------------------------------------------
    # False negatives
    # ---------------------------------------------------------

    false_negatives = data[
        data["false_negative"]
    ]

    print("\n" + "=" * 65)
    print("FALSE NEGATIVE CASES")
    print("=" * 65)

    if len(false_negatives) == 0:

        print("None")

    else:

        print(
            false_negatives[
                [
                    "transaction_id",
                    "invoice_id",
                    "actual_invoice_id",
                    "xgb_probability",
                    "actual_status",
                    "decision",
                    "decision_reason"
                ]
            ].to_string(index=False)
        )

    # ---------------------------------------------------------
    # Save evaluation
    # ---------------------------------------------------------

    output_file = "hybrid_v8_evaluation.csv"

    data.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nDetailed evaluation saved to: "
        f"{output_file}"
    )

    print("\n" + "=" * 65)
    print("V8 EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()