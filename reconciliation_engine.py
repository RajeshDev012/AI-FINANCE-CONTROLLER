import time
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Set
from rapidfuzz import fuzz

# =============================================================================
# CONFIGURATION & HYPERPARAMETERS
# =============================================================================
CONFIG = {
    # Feature Weights (Must sum to 1.0)
    "WEIGHT_AMOUNT": 0.45,
    "WEIGHT_VENDOR": 0.30,
    "WEIGHT_DATE": 0.10,
    "WEIGHT_REF": 0.15,
    
    # Absolute Financial Safety Gate Constraints
    "MAX_ABSOLUTE_AMOUNT_DIFFERENCE": 1.00,  # Max ₹1.00 variance allowed for auto-match
    
    # Candidate Generation Constraints
    "MAX_DATE_DIFFERENCE_DAYS": 45,
    "MAX_AMOUNT_VARIANCE_PCT": 0.10,        # 10% max variance for candidate evaluation pool
}

# Regex patterns for normalization
SUFFIXES_PATTERN = re.compile(
    r'\b(pvt ltd|private limited|ltd|llp|inc|corporation|corp|solutions|enterprises|india)\b', 
    re.IGNORECASE
)
PREFIXES_PATTERN = re.compile(
    r'^(rtgs|neft|imps|upi|bank transfer|wire transfer|ach|ref|pay|payment)[\/\-\s]*', 
    re.IGNORECASE
)


# =============================================================================
# DATA LOADING & NORMALIZATION
# =============================================================================
def load_data(invoices_path: str, bank_txns_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load invoices and bank transactions datasets from CSV."""
    df_invoices = pd.read_csv(invoices_path)
    df_bank = pd.read_csv(bank_txns_path)
    
    # Process only CREDIT transactions for invoice reconciliation
    df_credit_bank = df_bank[df_bank["transaction_type"] == "CREDIT"].copy()
    return df_invoices, df_credit_bank


def normalize_text(text: Optional[str]) -> str:
    """Standardize corporate names and transaction descriptions."""
    if not text or pd.isna(text):
        return ""
    
    cleaned = str(text).lower()
    cleaned = PREFIXES_PATTERN.sub('', cleaned)
    cleaned = SUFFIXES_PATTERN.sub('', cleaned)
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# =============================================================================
# FEATURE SCORING FUNCTIONS
# =============================================================================
def calculate_amount_metrics(inv_amount: float, txn_amount: float) -> Tuple[float, float, float]:
    """
    Calculate amount difference metrics.
    Returns (amount_difference, amount_difference_percent, amount_score).
    """
    diff = abs(inv_amount - txn_amount)
    diff_pct = (diff / inv_amount) if inv_amount > 0 else 1.0
    
    if diff == 0:
        score = 100.0
    else:
        score = max(0.0, 100.0 * (1.0 - (diff_pct / CONFIG["MAX_AMOUNT_VARIANCE_PCT"])))
        
    return round(diff, 2), round(diff_pct, 4), round(score, 2)


def calculate_vendor_similarity(norm_vendor: str, norm_description: str) -> float:
    """Calculate vendor similarity using RapidFuzz string comparison."""
    if not norm_vendor or not norm_description:
        return 0.0
    
    token_score = fuzz.token_sort_ratio(norm_vendor, norm_description)
    partial_score = fuzz.partial_ratio(norm_vendor, norm_description)
    return round(max(token_score, partial_score), 2)


def calculate_date_similarity(inv_date_str: str, txn_date_str: str) -> Tuple[float, int]:
    """Calculate date similarity score (0-100) and difference in days."""
    d1 = datetime.strptime(inv_date_str, "%Y-%m-%d")
    d2 = datetime.strptime(txn_date_str, "%Y-%m-%d")
    days_diff = abs((d2 - d1).days)
    
    max_days = CONFIG["MAX_DATE_DIFFERENCE_DAYS"]
    if days_diff > max_days:
        return 0.0, days_diff
    
    score = max(0.0, 100.0 * (1.0 - (days_diff / max_days)))
    return round(score, 2), days_diff


def check_reference_match(inv_num: str, description: str, reference: Optional[str]) -> float:
    """Returns 100.0 if invoice number explicitly appears in bank description or reference."""
    clean_inv_num = str(inv_num).replace('/', '').replace('-', '').lower()
    clean_desc = str(description).replace('/', '').replace('-', '').lower()
    clean_ref = str(reference).replace('/', '').replace('-', '').lower() if reference else ""
    
    if clean_inv_num in clean_desc or (clean_ref and clean_inv_num in clean_ref):
        return 100.0
    return 0.0


def calculate_confidence(
    amt_score: float, 
    vendor_score: float, 
    date_score: float, 
    ref_score: float
) -> float:
    """Calculate aggregate weighted confidence score."""
    score = (
        amt_score * CONFIG["WEIGHT_AMOUNT"] +
        vendor_score * CONFIG["WEIGHT_VENDOR"] +
        date_score * CONFIG["WEIGHT_DATE"] +
        ref_score * CONFIG["WEIGHT_REF"]
    )
    return round(score, 2)


# =============================================================================
# CANDIDATE GENERATION & EVALUATION
# =============================================================================
def generate_candidates(
    txn: pd.Series, 
    df_invoices: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Scan invoice master and evaluate matching metrics for candidates."""
    candidates = []
    norm_desc = normalize_text(txn["description"])
    
    for _, inv in df_invoices.iterrows():
        amt_diff, amt_diff_pct, amt_score = calculate_amount_metrics(inv["invoice_amount"], txn["amount"])
        
        # Prune candidates exceeding max allowable variance
        if amt_diff_pct > CONFIG["MAX_AMOUNT_VARIANCE_PCT"]:
            continue
            
        norm_vendor = normalize_text(inv["vendor_name"])
        vendor_score = calculate_vendor_similarity(norm_vendor, norm_desc)
        date_score, date_diff = calculate_date_similarity(inv["invoice_date"], txn["transaction_date"])
        ref_score = check_reference_match(inv["invoice_number"], txn["description"], txn.get("reference"))
        
        confidence = calculate_confidence(amt_score, vendor_score, date_score, ref_score)
        
        # Compute explicit priority tier for evidence-based sorting
        ref_match = (ref_score == 100.0)
        exact_amt = (amt_diff <= CONFIG["MAX_ABSOLUTE_AMOUNT_DIFFERENCE"])
        
        if ref_match and exact_amt:
            priority_tier = 1
        elif exact_amt and vendor_score >= 80:
            priority_tier = 2
        elif exact_amt and vendor_score >= 70 and date_diff <= 15:
            priority_tier = 3
        else:
            priority_tier = 4
        
        candidates.append({
            "invoice_id": inv["invoice_id"],
            "confidence": confidence,
            "priority_tier": priority_tier,
            "amount_difference": amt_diff,
            "amount_difference_percent": amt_diff_pct,
            "date_difference_days": date_diff,
            "vendor_similarity": vendor_score,
            "reference_match": ref_match,
            "amount_score": amt_score
        })
        
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates


# =============================================================================
# SAFE MATCH RULE & RECONCILIATION CORE
# =============================================================================
def is_safe_automatic_match(
    candidate: Dict[str, Any], 
    confidence_margin: float, 
    used_invoice_ids: Set[str]
) -> Tuple[bool, str]:
    """
    Evaluates multi-tier safe matching conditions and absolute safety gates.
    """
    inv_id = candidate["invoice_id"]
    amt_diff = candidate["amount_difference"]
    vendor_sim = candidate["vendor_similarity"]
    date_diff = candidate["date_difference_days"]
    conf = candidate["confidence"]
    ref_match = candidate["reference_match"]
    
    # -------------------------------------------------------------------------
    # ABSOLUTE SAFETY GATES (High Priority Intercepts)
    # -------------------------------------------------------------------------
    # 1. Absolute Amount Safety Gate
    if amt_diff > CONFIG["MAX_ABSOLUTE_AMOUNT_DIFFERENCE"]:
        return False, "Amount mismatch prevents automatic reconciliation"
    
    # 2. Duplicate Assignment Safety Gate
    if inv_id in used_invoice_ids:
        return False, "Invoice already assigned; possible duplicate payment"
    
    # 3. Ambiguity Gate (Candidate Margin)
    if confidence_margin < 8.0:
        return False, "Ambiguous: competing invoice candidates have similar evidence"
    
    # -------------------------------------------------------------------------
    # MULTI-CASE MATCH CRITERIA
    # -------------------------------------------------------------------------
    # CASE A: Explicit Invoice Reference Match + Exact Amount
    if ref_match and amt_diff <= 1.00 and confidence_margin >= 8.0:
        return True, "Invoice reference matched + exact amount + unambiguous candidate"
    
    # CASE B: Strong Vendor + Close Date + Exact Amount
    if (amt_diff <= 1.00 and vendor_sim >= 80.0 and date_diff <= 15 and 
        conf >= 82.0 and confidence_margin >= 8.0):
        return True, "Exact amount + strong vendor similarity + close date"
    
    # CASE C: Moderate Vendor + High Confidence & Margin + Exact Amount
    if (amt_diff <= 1.00 and vendor_sim >= 70.0 and date_diff <= 15 and 
        conf >= 88.0 and confidence_margin >= 15.0):
        return True, "Exact amount + strong evidence + large candidate margin"
    
    # Default Fallback for Unmet Conditions
    if vendor_sim < 70.0:
        return False, "Vendor/date evidence insufficient for automatic match"
    
    return False, "Vendor/date evidence insufficient for automatic match"


def run_reconciliation(invoices_path: str, bank_txns_path: str) -> pd.DataFrame:
    """Executes V4 reconciliation algorithm."""
    start_time = time.time()
    
    df_invoices, df_credit_bank = load_data(invoices_path, bank_txns_path)
    
    # Step 1: Candidate Generation
    txn_candidates_map = []
    for _, txn in df_credit_bank.iterrows():
        candidates = generate_candidates(txn, df_invoices)
        
        if candidates:
            best = candidates[0]
            sec_score = candidates[1]["confidence"] if len(candidates) > 1 else 0.0
            margin = best["confidence"] - sec_score
        else:
            best = None
            sec_score = 0.0
            margin = 0.0
            
        txn_candidates_map.append({
            "txn": txn,
            "candidates": candidates,
            "best": best,
            "top_confidence": best["confidence"] if best else 0.0,
            "priority_tier": best["priority_tier"] if best else 99,
            "second_best_score": sec_score,
            "confidence_margin": round(margin, 2)
        })
    
    # Step 2: Processing Order Priority (Tier first, then top confidence score)
    txn_candidates_map.sort(key=lambda x: (x["priority_tier"], -x["top_confidence"]))
    
    # Step 3: Sequential Allocation & Exception Handling
    used_invoice_ids: Set[str] = set()
    results = []

    for item in txn_candidates_map:
        txn = item["txn"]
        best = item["best"]
        sec_score = item["second_best_score"]
        margin = item["confidence_margin"]
        t_id = txn["transaction_id"]
        
        if not best:
            results.append({
                "transaction_id": t_id,
                "predicted_invoice_id": None,
                "confidence": 0.0,
                "decision": "UNMATCHED",
                "amount_difference": None,
                "amount_difference_percent": None,
                "date_difference_days": None,
                "vendor_similarity": 0.0,
                "reference_match": False,
                "second_best_score": 0.0,
                "confidence_margin": 0.0,
                "reason": "No sufficiently strong invoice candidate found"
            })
            continue

        target_inv_id = best["invoice_id"]
        
        # Evaluate Safe Automatic Match Rules
        is_match, reason = is_safe_automatic_match(best, margin, used_invoice_ids)
        
        if is_match:
            decision = "MATCH"
            used_invoice_ids.add(target_inv_id)  # Lock invoice immediately upon match
        else:
            decision = "REVIEW"

        results.append({
            "transaction_id": t_id,
            "predicted_invoice_id": target_inv_id,
            "confidence": best["confidence"],
            "decision": decision,
            "amount_difference": best["amount_difference"],
            "amount_difference_percent": round(best["amount_difference_percent"] * 100, 2),
            "date_difference_days": best["date_difference_days"],
            "vendor_similarity": best["vendor_similarity"],
            "reference_match": best["reference_match"],
            "second_best_score": sec_score,
            "confidence_margin": margin,
            "reason": reason
        })

    # Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv("reconciliation_results.csv", index=False)
    
    elapsed_time = round(time.time() - start_time, 2)
    
    # -------------------------------------------------------------------------
    # METRICS SUMMARY REPORT
    # -------------------------------------------------------------------------
    total_invoices = len(df_invoices)
    total_bank_txns = len(pd.read_csv(bank_txns_path))
    credit_txns = len(df_credit_bank)
    
    matches = df_results[df_results["decision"] == "MATCH"]
    match_count = len(matches)
    review_count = len(df_results[df_results["decision"] == "REVIEW"])
    unmatched_count = len(df_results[df_results["decision"] == "UNMATCHED"])
    
    match_rate = (match_count / credit_txns) * 100 if credit_txns > 0 else 0.0
    exception_rate = ((review_count + unmatched_count) / credit_txns) * 100 if credit_txns > 0 else 0.0
    avg_confidence = df_results["confidence"].mean()
    avg_match_confidence = matches["confidence"].mean() if match_count > 0 else 0.0

    print("\n==================================================")
    print("AI FINANCE CONTROLLER V4 — RECONCILIATION SUMMARY")
    print("==================================================")
    print(f"Total invoices:                  {total_invoices}")
    print(f"Total bank transactions:         {total_bank_txns}")
    print(f"CREDIT transactions processed:   {credit_txns}")
    print("--------------------------------------------------")
    print(f"MATCH:                           {match_count}")
    print(f"REVIEW:                          {review_count}")
    print(f"UNMATCHED:                       {unmatched_count}")
    print("--------------------------------------------------")
    print(f"Automatic match rate:            {match_rate:.2f}%")
    print(f"Exception rate:                  {exception_rate:.2f}%")
    print(f"Average confidence:              {avg_confidence:.2f}%")
    print(f"Average confidence (MATCH):      {avg_match_confidence:.2f}%")
    print(f"Processing time:                 {elapsed_time:.2f} seconds")
    print("==================================================\n")

    return df_results


if __name__ == "__main__":
    run_reconciliation("invoices.csv", "bank_transactions.csv")