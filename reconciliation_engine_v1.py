import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from datetime import datetime
import time
import re


# ============================================================
# CONFIGURATION
# ============================================================

MATCH_THRESHOLD = 90
REVIEW_THRESHOLD = 70

WEIGHT_REFERENCE = 0.20
WEIGHT_AMOUNT = 0.45
WEIGHT_VENDOR = 0.25
WEIGHT_DATE = 0.10


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    invoices = pd.read_csv("invoices.csv")
    transactions = pd.read_csv("bank_transactions.csv")

    required_invoice_columns = [
        "invoice_id",
        "invoice_date",
        "vendor_name",
        "invoice_number",
        "invoice_amount",
    ]

    required_transaction_columns = [
        "transaction_id",
        "transaction_date",
        "description",
        "amount",
    ]

    for column in required_invoice_columns:
        if column not in invoices.columns:
            raise ValueError(
                f"Missing column in invoices.csv: {column}"
            )

    for column in required_transaction_columns:
        if column not in transactions.columns:
            raise ValueError(
                f"Missing column in bank_transactions.csv: {column}"
            )

    invoices["invoice_date"] = pd.to_datetime(
        invoices["invoice_date"]
    )

    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"]
    )

    return invoices, transactions


# ============================================================
# 2. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize vendor names and transaction descriptions.

    Example:
        'CloudNova Technologies Pvt Ltd'
        becomes:
        'cloudnova technologies'
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

        # Remove common payment-method prefixes
    payment_terms = [
        "bank transfer",
        "rtgs",
        "neft",
        "imps",
        "upi",
        "ach",
        "wire transfer",
        "eft",
        "fast payment",
        "instant payment",
        "sepa",
        "bacs",
        "chaps",
        "fedwire",
        "chips",
        "credit card",
        "debit card",
        "prepaid card",
        "e-wallet",
        "digital wallet",
        "netbanking",
        "internet banking",
        "mobile banking",
        "cash deposit" ,
        "demand draft",
        "cheque",
        "ecs",
        "naach",
        "swift transfer",
        "crypto",
        "virtual currency",
        "cbdc",
        "fednow",
        "rtp",
        "sct inst",
        "paynow",
        "pix",
        "spei",
        "fps",
        "target2",
        "mepps",
        "ach dr",
        "ach cr",
        "sepa credit",
        "sepa direct debit",
        "ecs dr",
        "ecs cr",
        "nach dr",
        "nach cr",
        "ppo",
        "ppd",
        "ccd",
        "ctx",
        "swift payment",
        "remittance",
        "international wire",
        "interac",
        "mt103",
        "mt202",
        "mt940",
        "pacs.008",
        "pacs.009",
        "pain.001",
        "camt.053",
        "pos purchase",
        "pos txn",
        "contactless",
        "chip & pin",
        "card swipe",
        "stripe transfer",
        "razorpay",
        "paytm psp",
        "paypal payout",
        "adyen",
        "billdesk",
        "ccavenue",
        "square inc",
        "qr pay",
        "p2p payment",
        "online transfer",
        "direct deposit",
        "direct debit",
        "cash dep",
        "cdm deposit",
        "dd issue",
        "cheque trf",
        "chk deposit",
        "bankers cheque",
        "atm cash",
        "atw",
        "salary credit",
        "sal cr",
        "payroll",
        "disbursement",
        "stipend",
        "reimbursement",
        "allowance",
        "vendor payout",
        "e-mandate",
        "standing order",
        "auto-debit",
        "upi auto-pay",
        "rev",
        "reversal",
        "chargeback",
        "refund",
        "rtn",
        "returned",
        "bounced chk",
        "nsf",
        "int credit",
        "int debit",
        "service charge",
        "bank charges",
        "forex fee",
        "markup fee",
        "cbdc transfer",
        "e-rupee",
        "digital dollar",
        "usdt",
        "usdc",
        "on-ramp",
        "off-ramp"

        
    ]

    for term in payment_terms:
        text = text.replace(term, " ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove common business suffixes
    suffixes = [
        "private limited",
        "pvt ltd",
        "pvt",
        "limited",
        "ltd",
        "llp",
        "inc",
        "corporation",
        "corp",
    ]

    for suffix in suffixes:
        text = text.replace(suffix, " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 3. AMOUNT SIMILARITY
# ============================================================

def calculate_amount_similarity(invoice_amount, transaction_amount):
    """
    Return amount similarity from 0 to 100.
    """

    try:
        invoice_amount = float(invoice_amount)
        transaction_amount = float(transaction_amount)
    except (ValueError, TypeError):
        return 0.0

    if invoice_amount == transaction_amount:
        return 100.0

    if invoice_amount <= 0:
        return 0.0

    difference = abs(invoice_amount - transaction_amount)

    percentage_difference = (
        difference / invoice_amount
    ) * 100

    if percentage_difference <= 1:
        return 95.0

    if percentage_difference <= 2:
        return 90.0

    if percentage_difference <= 5:
        return 75.0

    if percentage_difference <= 10:
        return 50.0

    if percentage_difference <= 20:
        return 25.0

    return 0.0


# ============================================================
# 4. DATE SIMILARITY
# ============================================================

def calculate_date_similarity(invoice_date, transaction_date):
    """
    Calculate similarity based on days between invoice
    and bank transaction dates.
    """

    difference = abs(
        (transaction_date - invoice_date).days
    )

    if difference == 0:
        return 100.0

    if difference == 1:
        return 90.0

    if difference <= 3:
        return 75.0

    if difference <= 7:
        return 50.0

    if difference <= 14:
        return 25.0

    return 0.0


# ============================================================
# 5. VENDOR SIMILARITY
# ============================================================

def calculate_vendor_similarity(
    vendor_name,
    transaction_description
):
    """
    Compare invoice vendor name with bank transaction
    description using fuzzy matching.
    """

    vendor = normalize_text(vendor_name)
    description = normalize_text(transaction_description)

    if not vendor or not description:
        return 0.0

    # WRatio handles different word arrangements reasonably well
    score = fuzz.WRatio(vendor, description)

    return float(score)


# ============================================================
# 6. REFERENCE MATCH
# ============================================================

def check_reference_match(
    invoice_number,
    description
):
    """
    Check whether invoice number appears inside
    the bank transaction description.
    """

    if pd.isna(invoice_number) or pd.isna(description):
        return False

    invoice_number = str(invoice_number).lower().strip()
    description = str(description).lower()

    if not invoice_number:
        return False

    return invoice_number in description


# ============================================================
# 7. GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    transaction,
    invoices
):
    """
    Find reasonable invoice candidates for a transaction.

    We use amount/date/vendor information to reduce
    unnecessary comparisons.
    """

    transaction_amount = float(transaction["amount"])
    transaction_date = transaction["transaction_date"]
    description = transaction["description"]

    candidates = []

    for _, invoice in invoices.iterrows():

        amount_similarity = calculate_amount_similarity(
            invoice["invoice_amount"],
            transaction_amount
        )

        date_similarity = calculate_date_similarity(
            invoice["invoice_date"],
            transaction_date
        )

        vendor_similarity = calculate_vendor_similarity(
            invoice["vendor_name"],
            description
        )

        reference_match = check_reference_match(
            invoice["invoice_number"],
            description
        )

        # Candidate if at least one strong signal exists
        if (
            amount_similarity >= 50
            or vendor_similarity >= 60
            or date_similarity >= 75
            or reference_match
        ):
            candidates.append({
                "invoice": invoice,
                "amount_similarity": amount_similarity,
                "date_similarity": date_similarity,
                "vendor_similarity": vendor_similarity,
                "reference_match": reference_match,
            })

    return candidates


# ============================================================
# 8. CONFIDENCE SCORE
# ============================================================

def calculate_confidence(candidate):
    """
    Calculate reconciliation confidence.

    Strong financial evidence:
    - Exact amount
    - Strong vendor similarity
    - Close transaction date
    - Invoice reference when available

    Special rule:
    An exact amount + strong vendor match + close date
    is treated as a high-confidence reconciliation.
    """

    amount = candidate["amount_similarity"]
    vendor = candidate["vendor_similarity"]
    date = candidate["date_similarity"]
    reference = (
        100.0
        if candidate["reference_match"]
        else 0.0
    )

    # --------------------------------------------------------
    # Strong deterministic pattern
    # --------------------------------------------------------

    if (
        amount == 100
        and vendor >= 85
        and date >= 75
    ):
        confidence = 95.0

        # Extra evidence from invoice reference
        if candidate["reference_match"]:
            confidence = 99.0

        return confidence

    # --------------------------------------------------------
    # Normal weighted score
    # --------------------------------------------------------

    confidence = (
        reference * WEIGHT_REFERENCE
        + amount * WEIGHT_AMOUNT
        + vendor * WEIGHT_VENDOR
        + date * WEIGHT_DATE
    )

    return round(float(confidence), 2)


# ============================================================
# 9. DECISION
# ============================================================

def determine_decision(confidence):

    if confidence >= MATCH_THRESHOLD:
        return "MATCH"

    if confidence >= REVIEW_THRESHOLD:
        return "REVIEW"

    return "UNMATCHED"


# ============================================================
# 10. GENERATE EXPLANATION
# ============================================================

def generate_reason(candidate, confidence):

    reasons = []

    if candidate["reference_match"]:
        reasons.append("invoice reference matched")

    if candidate["amount_similarity"] == 100:
        reasons.append("exact amount")
    elif candidate["amount_similarity"] >= 75:
        reasons.append("close amount")
    else:
        reasons.append("amount difference")

    if candidate["vendor_similarity"] >= 90:
        reasons.append("strong vendor similarity")
    elif candidate["vendor_similarity"] >= 70:
        reasons.append("moderate vendor similarity")
    else:
        reasons.append("weak vendor similarity")

    if candidate["date_similarity"] >= 90:
        reasons.append("same/near date")
    elif candidate["date_similarity"] >= 50:
        reasons.append("date reasonably close")
    else:
        reasons.append("date difference")

    return " + ".join(reasons)


# ============================================================
# 11. RECONCILE ONE TRANSACTION
# ============================================================

def reconcile_transaction(
    transaction,
    invoices
):

    candidates = generate_candidates(
        transaction,
        invoices
    )

    if not candidates:

        return {
            "transaction_id": transaction["transaction_id"],
            "predicted_invoice_id": None,
            "confidence": 0,
            "decision": "UNMATCHED",
            "amount_difference": None,
            "date_difference_days": None,
            "vendor_similarity": 0,
            "reference_match": False,
            "reason": "No sufficiently strong invoice candidate found",
        }

    # Calculate confidence for every candidate
    for candidate in candidates:
        candidate["confidence"] = calculate_confidence(
            candidate
        )

    # Highest-confidence candidate
    best_candidate = max(
        candidates,
        key=lambda x: x["confidence"]
    )

    invoice = best_candidate["invoice"]

    confidence = best_candidate["confidence"]

    decision = determine_decision(confidence)

    amount_difference = abs(
        float(invoice["invoice_amount"])
        - float(transaction["amount"])
    )

    date_difference = abs(
        (
            transaction["transaction_date"]
            - invoice["invoice_date"]
        ).days
    )

    reason = generate_reason(
        best_candidate,
        confidence
    )

    return {
        "transaction_id": transaction["transaction_id"],
        "predicted_invoice_id": invoice["invoice_id"],
        "confidence": confidence,
        "decision": decision,
        "amount_difference": round(
            amount_difference,
            2
        ),
        "date_difference_days": date_difference,
        "vendor_similarity": round(
            best_candidate["vendor_similarity"],
            2
        ),
        "reference_match": best_candidate["reference_match"],
        "reason": reason,
    }


# ============================================================
# 12. RUN RECONCILIATION
# ============================================================

def run_reconciliation():

    start_time = time.time()

    invoices, transactions = load_data()

    # Only CREDIT transactions are relevant for incoming
    # payment reconciliation.
    if "transaction_type" in transactions.columns:
        transactions = transactions[
            transactions["transaction_type"]
            .astype(str)
            .str.upper()
            == "CREDIT"
        ].copy()

    results = []

    for _, transaction in transactions.iterrows():

        result = reconcile_transaction(
            transaction,
            invoices
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df.to_csv(
        "reconciliation_results.csv",
        index=False
    )

    elapsed_time = time.time() - start_time

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    match_count = (
        results_df["decision"] == "MATCH"
    ).sum()

    review_count = (
        results_df["decision"] == "REVIEW"
    ).sum()

    unmatched_count = (
        results_df["decision"] == "UNMATCHED"
    ).sum()

    print()
    print("=" * 50)
    print("AI FINANCE CONTROLLER")
    print("RECONCILIATION SUMMARY")
    print("=" * 50)

    print(f"Total invoices:      {len(invoices)}")
    print(f"Bank transactions:   {len(transactions)}")
    print(f"Processed:            {len(results_df)}")

    print()
    print(f"MATCH:                {match_count}")
    print(f"REVIEW:               {review_count}")
    print(f"UNMATCHED:            {unmatched_count}")

    if len(results_df) > 0:

        match_rate = (
            match_count
            / len(results_df)
        ) * 100

        exception_rate = (
            (review_count + unmatched_count)
            / len(results_df)
        ) * 100

        print()
        print(
            f"Automatic match rate: {match_rate:.2f}%"
        )

        print(
            f"Exception rate:        {exception_rate:.2f}%"
        )

    print()
    print(
        f"Processing time:       {elapsed_time:.4f} seconds"
    )

    print("=" * 50)

    print()
    print(
        "Results saved to: reconciliation_results.csv"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_reconciliation()