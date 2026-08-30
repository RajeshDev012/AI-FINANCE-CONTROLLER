import streamlit as st
import pandas as pd


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Finance Controller",
    page_icon="💶",
    layout="wide"
)


# ============================================================
# FILES
# ============================================================

RESULT_FILE = "hybrid_v8_results.csv"
EVALUATION_FILE = "hybrid_v8_evaluation.csv"


# ============================================================
# LOAD RESULTS
# ============================================================

try:
    df = pd.read_csv(RESULT_FILE)
except FileNotFoundError:
    st.error(
        "hybrid_v8_results.csv not found.\n\n"
        "Run: python hybrid_reconciliation_v8.py"
    )
    st.stop()


# ============================================================
# LOAD EVALUATION IF AVAILABLE
# ============================================================

try:
    evaluation = pd.read_csv(EVALUATION_FILE)
except FileNotFoundError:
    evaluation = None


# ============================================================
# HEADER
# ============================================================

st.title("💶 AI Finance Controller")

st.subheader(
    "Hybrid XGBoost Financial Reconciliation System"
)

st.write(
    "Automatically matches bank transactions with invoices "
    "using machine learning and financial safety rules."
)


# ============================================================
# BASIC METRICS
# ============================================================

total = len(df)

matches = int(
    (df["decision"] == "MATCH").sum()
)

reviews = int(
    (df["decision"] == "REVIEW").sum()
)

match_rate = (
    matches / total * 100
    if total
    else 0
)

avg_probability = (
    df["xgb_probability"].mean() * 100
    if total
    else 0
)


# ============================================================
# HIGH CONFIDENCE
# ============================================================

high_confidence = int(
    (
        (df["decision"] == "MATCH")
        &
        (df["xgb_probability"] >= 0.90)
    ).sum()
)


# ============================================================
# TOP KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Transactions",
        total
    )

with col2:
    st.metric(
        "Automatic Matches",
        matches
    )

with col3:
    st.metric(
        "Review Required",
        reviews
    )

with col4:
    st.metric(
        "Match Rate",
        f"{match_rate:.2f}%"
    )


# ============================================================
# SECONDARY KPI
# ============================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average XGBoost Probability",
        f"{avg_probability:.2f}%"
    )

with col2:
    st.metric(
        "High Confidence Matches",
        high_confidence
    )

with col3:
    st.metric(
        "Transactions for Review",
        reviews
    )


# ============================================================
# EVALUATION
# ============================================================

if evaluation is not None:

    st.divider()

    st.subheader(
        "🎯 Model Evaluation"
    )

    # Evaluation CSV contains one row per transaction.
    # Calculate metrics from the boolean columns.

    tp = int(
        evaluation["true_positive"].sum()
    )

    fp = int(
        evaluation["false_positive"].sum()
    )

    fn = int(
        evaluation["false_negative"].sum()
    )

    tn = int(
        evaluation["true_negative"].sum()
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Precision",
            f"{precision:.2%}"
        )

    with col2:
        st.metric(
            "Recall",
            f"{recall:.2%}"
        )

    with col3:
        st.metric(
            "F1 Score",
            f"{f1:.2%}"
        )

    with col4:
        st.metric(
            "False Negatives",
            fn
        )

    st.caption(
        "Evaluation based on the ground-truth dataset. "
        "Ground truth is not used by the production engine."
    )

    # Confusion matrix

    st.write("### Confusion Matrix")

    confusion = pd.DataFrame(
        {
            "Predicted MATCH": [
                tp,
                fp
            ],
            "Predicted REVIEW": [
                fn,
                tn
            ]
        },
        index=[
            "Actual MATCH",
            "Actual NON-MATCH"
        ]
    )

    st.dataframe(
        confusion,
        use_container_width=True
    )


# ============================================================
# RECONCILIATION OVERVIEW
# ============================================================

st.divider()

st.subheader(
    "📊 Reconciliation Overview"
)

col1, col2 = st.columns(2)


with col1:

    decision_counts = (
        df["decision"]
        .value_counts()
    )

    st.bar_chart(
        decision_counts
    )


with col2:

    st.write(
        "### Decision Distribution"
    )

    distribution = pd.DataFrame(
        {
            "Decision":
                decision_counts.index,
            "Count":
                decision_counts.values
        }
    )

    st.dataframe(
        distribution,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SEARCH
# ============================================================

st.divider()

st.subheader(
    "🔎 Transaction Search"
)

col1, col2, col3 = st.columns(3)


with col1:

    decision_filter = st.selectbox(
        "Decision",
        [
            "ALL",
            "MATCH",
            "REVIEW"
        ]
    )


with col2:

    min_probability = st.slider(
        "Minimum XGBoost Probability",
        0.0,
        1.0,
        0.0,
        0.01
    )


with col3:

    search = st.text_input(
        "Transaction / Invoice ID"
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = df.copy()


if decision_filter != "ALL":

    filtered = filtered[
        filtered["decision"]
        == decision_filter
    ]


filtered = filtered[
    filtered["xgb_probability"]
    >= min_probability
]


if search:

    search = search.upper()

    filtered = filtered[
        filtered[
            "transaction_id"
        ]
        .astype(str)
        .str.upper()
        .str.contains(search)
        |
        filtered[
            "invoice_id"
        ]
        .astype(str)
        .str.upper()
        .str.contains(search)
    ]


# ============================================================
# RESULTS
# ============================================================

st.subheader(
    f"📋 Reconciliation Results ({len(filtered)})"
)


display_columns = [

    "transaction_id",

    "invoice_id",

    "xgb_probability",

    "model_margin",

    "amount_difference_percent",

    "vendor_similarity",

    "date_difference_days",

    "reference_match",

    "decision",

    "decision_reason"

]


display_columns = [
    col
    for col in display_columns
    if col in filtered.columns
]


display_df = filtered[
    display_columns
].copy()


# Convert probability to percentage

if "xgb_probability" in display_df.columns:

    display_df[
        "xgb_probability"
    ] = (
        display_df[
            "xgb_probability"
        ] * 100
    ).round(2)


# Rename columns

display_df = display_df.rename(
    columns={

        "transaction_id":
            "Transaction ID",

        "invoice_id":
            "Invoice ID",

        "xgb_probability":
            "XGBoost %",

        "model_margin":
            "Model Margin",

        "amount_difference_percent":
            "Amount Difference %",

        "vendor_similarity":
            "Vendor Similarity",

        "date_difference_days":
            "Date Difference Days",

        "reference_match":
            "Reference Match",

        "decision":
            "Decision",

        "decision_reason":
            "Decision Reason"
    }
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD
# ============================================================

st.download_button(
    label="⬇️ Download Filtered Results",
    data=display_df.to_csv(
        index=False
    ).encode("utf-8"),
    file_name="reconciliation_filtered_results.csv",
    mime="text/csv"
)


# ============================================================
# REVIEW SECTION
# ============================================================

st.divider()

st.subheader(
    "⚠️ Transactions Requiring Review"
)


review_df = df[
    df["decision"] == "REVIEW"
].copy()


if len(review_df) == 0:

    st.success(
        "No transactions currently require review."
    )

else:

    review_display = review_df[
        [
            "transaction_id",
            "invoice_id",
            "xgb_probability",
            "model_margin",
            "amount_difference_percent",
            "decision_reason",
            "explanation"
        ]
    ].copy()

    review_display[
        "xgb_probability"
    ] = (
        review_display[
            "xgb_probability"
        ] * 100
    ).round(2)

    st.dataframe(
        review_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🤖 Model Information"
)

col1, col2 = st.columns(2)


with col1:

    st.write(
        "**Model:** XGBoost V7"
    )

    st.write(
        "**Architecture:** Hybrid ML + Safety Controller"
    )

    st.write(
        "**Automatic decision:** MATCH"
    )

    st.write(
        "**Exception decision:** REVIEW"
    )


with col2:

    st.write(
        "**XGBoost threshold:** 0.50"
    )

    st.write(
        "**Minimum model margin:** 0.08"
    )

    st.write(
        "**Amount tolerance:** 5%"
    )

    st.write(
        "**Ground truth:** Used only for evaluation"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Finance Controller V8 | "
    "Hybrid XGBoost Financial Reconciliation"
)