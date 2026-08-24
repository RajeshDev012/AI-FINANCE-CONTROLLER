import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Fix random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def generate_all_data():
    # -------------------------------------------------------------------------
    # 1. GENERATE INVOICES (500 records)
    # -------------------------------------------------------------------------
    vendors = [
        ("VEND-101", "Cloudnova Technologies Pvt Ltd"),
        ("VEND-102", "Metro Logistics Solutions"),
        ("VEND-103", "Apex Infraworks India"),
        ("VEND-104", "Zephyr Digital Media"),
        ("VEND-105", "Trident Supply Chain Solutions"),
        ("VEND-106", "Kaveri Enterprises Solutions"),
        ("VEND-107", "Omniware Systems India"),
        ("VEND-108", "Bharat Fuel Services"),
        ("VEND-109", "Sterling Office Supplies"),
        ("VEND-110", "Vanguard Security Services")
    ]
    
    categories = [
        "IT Services", "Logistics", "Construction", "Marketing",
        "Supply Chain", "General Operations", "Software", "Fuel & Energy",
        "Office Supplies", "Facility & Security"
    ]

    base_date = datetime(2026, 8, 24)
    start_date = base_date - timedelta(days=365)

    invoices = []
    for i in range(1, 501):
        inv_id = f"INV-{i:04d}"
        inv_num = f"INV/2025-26/{1000 + i}"
        
        # Pick vendor & matching category index
        v_idx = random.randint(0, len(vendors) - 1)
        vendor_id, vendor_name = vendors[v_idx]
        category = categories[v_idx]
        
        # Amount: ₹500 to ₹500,000 (rounded to nearest 50)
        amount = float(random.randint(10, 10000) * 50)
        
        # Date within last 365 days
        days_offset = random.randint(0, 360)
        inv_date = start_date + timedelta(days=days_offset)
        due_date = inv_date + timedelta(days=random.choice([15, 30, 45]))

        invoices.append({
            "invoice_id": inv_id,
            "invoice_date": inv_date.strftime("%Y-%m-%d"),
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "invoice_number": inv_num,
            "invoice_amount": amount,
            "currency": "INR",
            "payment_due_date": due_date.strftime("%Y-%m-%d"),
            "invoice_status": "UNPAID",
            "category": category
        })

    df_invoices = pd.DataFrame(invoices)

    # -------------------------------------------------------------------------
    # 2. SCENARIO SPLIT & GENERATION DESIGN
    # Target Target counts out of 500 invoices:
    # ~375 Normal Match (75%)
    # ~40 Amount Mismatch (8%)
    # ~25 Ambiguous Cases (5% - 25 invoices yielding ~25 ambiguous transactions)
    # ~20 Duplicate Cases (4% - 20 invoices receiving 2 transactions each)
    # ~25 Missing Payments (5% - 25 invoices receiving 0 bank transactions)
    # -------------------------------------------------------------------------
    
    shuffled_inv_indices = list(range(500))
    random.shuffle(shuffled_inv_indices)

    idx_match = shuffled_inv_indices[0:375]
    idx_amt_mismatch = shuffled_inv_indices[375:415]
    idx_ambiguous = shuffled_inv_indices[415:440]
    idx_duplicate = shuffled_inv_indices[440:460]
    idx_missing = shuffled_inv_indices[460:500]  # 40 missing payments

    bank_txns = []
    ground_truths = []
    missing_payments = []
    
    txn_counter = 6000

    def get_txn_id():
        nonlocal txn_counter
        txn_counter += 1
        return f"TXN-{txn_counter}"

    def generate_narrative(vendor_name, inv_num, prefix=""):
        patterns = [
            f"NEFT/{prefix}{vendor_name.upper()}",
            f"RTGS/{vendor_name.upper()[:15]}/REF-{random.randint(100000,999999)}",
            f"IMPS/{vendor_name.split()[0].upper()}/{inv_num}",
            f"UPI/{vendor_name.upper()[:12]}@okbank/PAYMENT",
            f"BANK TRANSFER {vendor_name.upper()}"
        ]
        return random.choice(patterns)

    # A. NORMAL MATCH SCENARIO (375)
    for idx in idx_match:
        inv = df_invoices.iloc[idx]
        t_id = get_txn_id()
        p_date = datetime.strptime(inv["invoice_date"], "%Y-%m-%d") + timedelta(days=random.randint(1, 20))
        
        bank_txns.append({
            "transaction_id": t_id,
            "transaction_date": p_date.strftime("%Y-%m-%d"),
            "description": generate_narrative(inv["vendor_name"], inv["invoice_number"]),
            "amount": inv["invoice_amount"],
            "currency": "INR",
            "transaction_type": "CREDIT",
            "reference": f"REF-{t_id}",
            "bank_status": "SETTLED"
        })
        ground_truths.append({
            "transaction_id": t_id,
            "actual_invoice_id": inv["invoice_id"],
            "actual_status": "MATCH",
            "actual_reason": "Exact amount, vendor, and single valid mapping."
        })

    # B. AMOUNT MISMATCH SCENARIO (40)
    for idx in idx_amt_mismatch:
        inv = df_invoices.iloc[idx]
        t_id = get_txn_id()
        p_date = datetime.strptime(inv["invoice_date"], "%Y-%m-%d") + timedelta(days=random.randint(1, 20))
        
        # Introduce deliberate delta (underpayment/overpayment or minor fee deduction)
        diff = random.choice([-500.0, -150.0, -50.0, 100.0, 500.0])
        actual_paid = max(100.0, inv["invoice_amount"] + diff)

        bank_txns.append({
            "transaction_id": t_id,
            "transaction_date": p_date.strftime("%Y-%m-%d"),
            "description": generate_narrative(inv["vendor_name"], inv["invoice_number"]),
            "amount": actual_paid,
            "currency": "INR",
            "transaction_type": "CREDIT",
            "reference": f"REF-{t_id}",
            "bank_status": "SETTLED"
        })
        ground_truths.append({
            "transaction_id": t_id,
            "actual_invoice_id": inv["invoice_id"],
            "actual_status": "AMOUNT_MISMATCH",
            "actual_reason": f"Bank txn amount ({actual_paid}) differs from invoice amount ({inv['invoice_amount']})."
        })

    # C. AMBIGUOUS CASES SCENARIO (25)
    # Make two invoices identical in vendor, date, and amount to create true structural ambiguity
    for i in range(0, len(idx_ambiguous), 2):
        if i + 1 >= len(idx_ambiguous):
            break
        idx1 = idx_ambiguous[i]
        idx2 = idx_ambiguous[i+1]
        
        # Override invoice 2 to match invoice 1 exactly in vendor, amount, and date
        inv1 = df_invoices.iloc[idx1]
        df_invoices.at[idx2, "vendor_id"] = inv1["vendor_id"]
        df_invoices.at[idx2, "vendor_name"] = inv1["vendor_name"]
        df_invoices.at[idx2, "invoice_amount"] = inv1["invoice_amount"]
        df_invoices.at[idx2, "invoice_date"] = inv1["invoice_date"]
        df_invoices.at[idx2, "category"] = inv1["category"]
        
        inv2 = df_invoices.iloc[idx2]
        
        # Create bank transaction targeting this identical pair
        t_id = get_txn_id()
        p_date = datetime.strptime(inv1["invoice_date"], "%Y-%m-%d") + timedelta(days=random.randint(1, 5))
        
        bank_txns.append({
            "transaction_id": t_id,
            "transaction_date": p_date.strftime("%Y-%m-%d"),
            "description": f"NEFT/{inv1['vendor_name'].upper()}/GENERIC PAY",
            "amount": inv1["invoice_amount"],
            "currency": "INR",
            "transaction_type": "CREDIT",
            "reference": f"REF-{t_id}",
            "bank_status": "SETTLED"
        })
        ground_truths.append({
            "transaction_id": t_id,
            "actual_invoice_id": inv1["invoice_id"],  # Internal ground truth pointer
            "actual_status": "AMBIGUOUS",
            "actual_reason": f"Multiple identical invoices ({inv1['invoice_id']}, {inv2['invoice_id']}) exist for vendor {inv1['vendor_name']} and amount {inv1['invoice_amount']}."
        })

    # D. DUPLICATE PAYMENT SCENARIO (20 invoices -> 40 bank transactions)
    for idx in idx_duplicate:
        inv = df_invoices.iloc[idx]
        p_date = datetime.strptime(inv["invoice_date"], "%Y-%m-%d") + timedelta(days=random.randint(1, 10))
        
        # Txn 1: Legitimate initial payment (MATCH)
        t_id1 = get_txn_id()
        bank_txns.append({
            "transaction_id": t_id1,
            "transaction_date": p_date.strftime("%Y-%m-%d"),
            "description": generate_narrative(inv["vendor_name"], inv["invoice_number"], "ORIG-"),
            "amount": inv["invoice_amount"],
            "currency": "INR",
            "transaction_type": "CREDIT",
            "reference": f"REF-{t_id1}",
            "bank_status": "SETTLED"
        })
        ground_truths.append({
            "transaction_id": t_id1,
            "actual_invoice_id": inv["invoice_id"],
            "actual_status": "MATCH",
            "actual_reason": "Original valid transaction matching invoice."
        })
        
        # Txn 2: Duplicate accidental payment (DUPLICATE)
        t_id2 = get_txn_id()
        dup_date = p_date + timedelta(days=random.randint(1, 5))
        bank_txns.append({
            "transaction_id": t_id2,
            "transaction_date": dup_date.strftime("%Y-%m-%d"),
            "description": generate_narrative(inv["vendor_name"], inv["invoice_number"], "DUP-"),
            "amount": inv["invoice_amount"],
            "currency": "INR",
            "transaction_type": "CREDIT",
            "reference": f"REF-{t_id2}",
            "bank_status": "SETTLED"
        })
        ground_truths.append({
            "transaction_id": t_id2,
            "actual_invoice_id": inv["invoice_id"],
            "actual_status": "DUPLICATE",
            "actual_reason": f"Second transaction paying already-settled invoice {inv['invoice_id']} (Primary TXN: {t_id1})."
        })

    # E. DEBIT TRANSACTIONS (15 records to add realistic bank environment noise)
    for _ in range(15):
        t_id = get_txn_id()
        d_date = start_date + timedelta(days=random.randint(0, 360))
        bank_txns.append({
            "transaction_id": t_id,
            "transaction_date": d_date.strftime("%Y-%m-%d"),
            "description": random.choice(["BANK CHARGES", "VENDOR REFUND OUTWARD", "GST TAX PAYMENT", "SALARY PAYROLL"]),
            "amount": float(random.randint(5, 500) * 100),
            "currency": "INR",
            "transaction_type": "DEBIT",
            "reference": f"REF-{t_id}",
            "bank_status": "SETTLED"
        })

    # F. MISSING PAYMENTS SCENARIO (40 invoices with zero bank transactions)
    for idx in idx_missing:
        inv = df_invoices.iloc[idx]
        missing_payments.append({
            "invoice_id": inv["invoice_id"],
            "invoice_number": inv["invoice_number"],
            "invoice_amount": inv["invoice_amount"],
            "invoice_date": inv["invoice_date"],
            "reason": "NO_PAYMENT_FOUND"
        })

    # Save DataFrames
    df_invoices.to_csv("invoices.csv", index=False)
    df_bank_txns = pd.DataFrame(bank_txns)
    df_bank_txns.to_csv("bank_transactions.csv", index=False)
    df_ground_truth = pd.DataFrame(ground_truths)
    df_ground_truth.to_csv("ground_truth.csv", index=False)
    df_missing = pd.DataFrame(missing_payments)
    df_missing.to_csv("missing_payments.csv", index=False)

    # -------------------------------------------------------------------------
    # 3. GENERATE DATA QUALITY REPORT
    # -------------------------------------------------------------------------
    credit_txns = df_bank_txns[df_bank_txns["transaction_type"] == "CREDIT"]
    debit_txns = df_bank_txns[df_bank_txns["transaction_type"] == "DEBIT"]
    
    report_data = [{
        "total_invoices": len(df_invoices),
        "total_credit_transactions": len(credit_txns),
        "total_debit_transactions": len(debit_txns),
        "normal_matches": len(df_ground_truth[df_ground_truth["actual_status"] == "MATCH"]),
        "amount_mismatches": len(df_ground_truth[df_ground_truth["actual_status"] == "AMOUNT_MISMATCH"]),
        "ambiguous_cases": len(df_ground_truth[df_ground_truth["actual_status"] == "AMBIGUOUS"]),
        "duplicate_payments": len(df_ground_truth[df_ground_truth["actual_status"] == "DUPLICATE"]),
        "missing_payments": len(df_missing),
        "unique_transaction_ids": df_bank_txns["transaction_id"].nunique(),
        "duplicate_transaction_ids": len(df_bank_txns) - df_bank_txns["transaction_id"].nunique(),
        "ground_truth_rows": len(df_ground_truth)
    }]
    
    pd.DataFrame(report_data).to_csv("data_quality_report.csv", index=False)

if __name__ == "__main__":
    generate_all_data()
    print("Dataset generated successfully.")