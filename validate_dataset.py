import pandas as pd
import sys
import os

def validate():
    print("Beginning dataset validation...")
    base_dir = "finance_dataset"
    
    invoices = pd.read_csv(os.path.join(base_dir, "invoices.csv"))
    txns = pd.read_csv(os.path.join(base_dir, "bank_transactions.csv"))
    gt = pd.read_csv(os.path.join(base_dir, "ground_truth.csv"))
    
    # 1. Unique ID checks
    assert invoices['invoice_id'].is_unique, "Duplicate invoice_ids found!"
    assert txns['transaction_id'].is_unique, "Duplicate transaction_ids found!"
    
    # 2. Null checks
    assert not invoices.isnull().any().any(), "Invoices contains null values!"
    assert not txns.isnull().any().any(), "Transactions contains null values!"
    assert not gt.isnull().any().any(), "Ground truth contains null values!"
    
    # 3. Numeric checks
    assert (invoices['invoice_amount'] > 0).all(), "Non-positive invoice amounts found!"
    assert (txns['amount'] > 0).all(), "Non-positive transaction amounts found!"
    
    # 4. Ground truth check
    valid_txns = set(txns['transaction_id']).union({"NONE"})
    assert gt['transaction_id'].isin(valid_txns).all(), "Ground truth contains unknown transaction_ids!"
    
    valid_invs = set(invoices['invoice_id'])
    assert gt['invoice_id'].isin(valid_invs).all(), "Ground truth contains unknown invoice_ids!"
    
    print("Dataset validation PASSED successfully!")
    print(f"Total Invoices Verified: {len(invoices)}")
    print(f"Total Transactions Verified: {len(txns)}")
    print(f"Total Mappings Verified: {len(gt)}")

if __name__ == "__main__":
    try:
        validate()
    except Exception as e:
        print(f"Validation FAILED: {str(e)}")
        sys.exit(1)
