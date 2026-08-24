import pandas as pd

def run_validation():
    errors = []

    # Load files
    try:
        invoices = pd.read_csv("invoices.csv")
        bank_txns = pd.read_csv("bank_transactions.csv")
        ground_truth = pd.read_csv("ground_truth.csv")
        missing_payments = pd.read_csv("missing_payments.csv")
        report = pd.read_csv("data_quality_report.csv")
    except Exception as e:
        print(f"FAILED TO LOAD FILES: {e}")
        return False

    # 1. invoices.csv has exactly 500 invoices
    if len(invoices) != 500:
        errors.append(f"Check 1 Failed: invoices.csv count is {len(invoices)}, expected 500.")

    # 2. invoice_id is unique
    if invoices["invoice_id"].nunique() != len(invoices):
        errors.append("Check 2 Failed: duplicate invoice_id values found in invoices.csv.")

    # 3. invoice_number is unique
    if invoices["invoice_number"].nunique() != len(invoices):
        errors.append("Check 3 Failed: duplicate invoice_number values found in invoices.csv.")

    # 4. transaction_id is unique in bank_transactions.csv
    if bank_txns["transaction_id"].nunique() != len(bank_txns):
        errors.append("Check 4 Failed: duplicate transaction_id values in bank_transactions.csv.")

    # 5. transaction_id is unique in ground_truth.csv
    if ground_truth["transaction_id"].nunique() != len(ground_truth):
        errors.append("Check 5 Failed: duplicate transaction_id values in ground_truth.csv.")

    # 6. Every CREDIT transaction has exactly one ground-truth row
    credit_txns = bank_txns[bank_txns["transaction_type"] == "CREDIT"]
    if len(credit_txns) != len(ground_truth):
        errors.append(f"Check 6 Failed: CREDIT transactions ({len(credit_txns)}) != ground truth rows ({len(ground_truth)}).")

    # 7. Every ground-truth transaction_id exists in bank_transactions.csv
    missing_gt_txns = set(ground_truth["transaction_id"]) - set(bank_txns["transaction_id"])
    if len(missing_gt_txns) > 0:
        errors.append(f"Check 7 Failed: {len(missing_gt_txns)} ground_truth transaction_ids missing from bank_transactions.csv.")

    # 8. Every MATCH invoice exists in invoices.csv
    match_invs = ground_truth[ground_truth["actual_status"] == "MATCH"]["actual_invoice_id"]
    missing_match_invs = set(match_invs) - set(invoices["invoice_id"])
    if len(missing_match_invs) > 0:
        errors.append(f"Check 8 Failed: Ground truth MATCH contains invalid invoice IDs: {missing_match_invs}")

    # 9. Every DUPLICATE invoice exists in invoices.csv
    dup_invs = ground_truth[ground_truth["actual_status"] == "DUPLICATE"]["actual_invoice_id"]
    missing_dup_invs = set(dup_invs) - set(invoices["invoice_id"])
    if len(missing_dup_invs) > 0:
        errors.append(f"Check 9 Failed: Ground truth DUPLICATE contains invalid invoice IDs: {missing_dup_invs}")

    # 10. missing_payments.csv contains invoices that have no bank transaction
    gt_matched_invs = set(ground_truth["actual_invoice_id"])
    missing_in_gt = set(missing_payments["invoice_id"]).intersection(gt_matched_invs)
    if len(missing_in_gt) > 0:
        errors.append(f"Check 10 Failed: missing_payments.csv contains invoice IDs present in ground truth: {missing_in_gt}")

    # 11. No missing payment has a fake transaction ID
    if "transaction_id" in missing_payments.columns:
        errors.append("Check 11 Failed: missing_payments.csv should not contain a transaction_id column.")

    # 12. Amounts are valid positive numbers
    if (invoices["invoice_amount"] <= 0).any() or (bank_txns["amount"] <= 0).any():
        errors.append("Check 12 Failed: Non-positive amounts detected.")

    # 13. Dates are valid
    try:
        pd.to_datetime(invoices["invoice_date"])
        pd.to_datetime(bank_txns["transaction_date"])
    except Exception as e:
        errors.append(f"Check 13 Failed: Invalid date formats detected: {e}")

    # 14. Ground-truth counts match actual generated data
    rep = report.iloc[0]
    if rep["ground_truth_rows"] != len(ground_truth) or rep["total_invoices"] != len(invoices):
        errors.append("Check 14 Failed: Data quality report counts do not match dataset files.")

    if errors:
        print("DATASET VALIDATION: FAILED")
        for err in errors:
            print(f"  - {err}")
        return False

    print("DATASET VALIDATION: PASSED")
    print(f"Total Invoices: {len(invoices)}")
    print(f"Total Bank Transactions: {len(bank_txns)} ({len(credit_txns)} CREDIT, {len(bank_txns) - len(credit_txns)} DEBIT)")
    print(f"Ground Truth Records: {len(ground_truth)}")
    print(f"  - MATCH: {rep['normal_matches']}")
    print(f"  - AMOUNT_MISMATCH: {rep['amount_mismatches']}")
    print(f"  - AMBIGUOUS: {rep['ambiguous_cases']}")
    print(f"  - DUPLICATE: {rep['duplicate_payments']}")
    print(f"Missing Payments: {len(missing_payments)}")
    return True

if __name__ == "__main__":
    run_validation()