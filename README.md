# 💶 AI Finance Controller

### Hybrid XGBoost Financial Reconciliation System

> An intelligent financial reconciliation system that automatically matches bank transactions with invoices using **XGBoost Machine Learning + deterministic financial safety rules**, while sending uncertain transactions for human review.

---

## 🚀 Project Overview

Financial reconciliation is the process of matching incoming bank transactions with their corresponding invoices.

Traditional reconciliation systems often depend on simple rules such as:

- Exact amount matching
- Vendor name matching
- Invoice reference matching
- Transaction date matching

These rules can work for simple cases, but real-world financial data contains:

- Different vendor naming formats
- Missing invoice references
- Date differences
- Amount discrepancies
- Duplicate payments
- Ambiguous candidates
- Missing payments

The **AI Finance Controller** addresses this problem using a hybrid approach.

Instead of relying only on machine learning or only on rules, the system combines both:

```text
                Bank Transaction
                       │
                       ▼
              Candidate Generation
                       │
                       ▼
              Feature Engineering
                       │
                       ▼
                 XGBoost Model
                       │
                       ▼
               Best Candidate
                       │
                       ▼
              Safety Controller
                  /     |      \
                 /      |       \
                ▼       ▼        ▼
             MATCH    REVIEW   UNMATCHED
                │       │
                │       ▼
                │   Human Review
                │
                ▼
             Audit Logset.

--Installation section..

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI_FINANCE_CONTROLLER

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

python hybrid_reconciliation_v8.py
python evaluate_v8.py
streamlit run app.py