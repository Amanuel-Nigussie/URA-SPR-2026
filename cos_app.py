import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("similarities.csv")


# -----------------------------------------------------------

st.title("CodeBERT Semantic Similarity Analysis")

st.write("""
This dashboard analyzes the semantic similarity between **correct and incorrect implementations**
using **CodeBERT embeddings and cosine similarity**.
""")


# -----------------------------------------------------------

df = load_data()

all_folds = sorted(df["fold"].unique())
all_splits = ["fit", "validate", "test"]

selected_folds = st.multiselect("Select folds", all_folds, default=all_folds)
selected_split = st.selectbox("Select split", all_splits)


# -----------------------------------------------------------

filtered = df[(df["fold"].isin(selected_folds)) & (df["split"] == selected_split)]


# -----------------------------------------------------------

st.header("Statistics per Fold")

rows = []

for fold in selected_folds:
    fold_df = filtered[filtered["fold"] == fold]["sim"]
    rows.append({
        "Fold":   fold,
        "Count":  len(fold_df),
        "Mean":   round(fold_df.mean(), 4),
        "Median": round(fold_df.median(), 4),
        "Std":    round(fold_df.std(), 4),
        "Min":    round(fold_df.min(), 4),
        "Max":    round(fold_df.max(), 4),
    })

st.dataframe(pd.DataFrame(rows).set_index("Fold"))


# -----------------------------------------------------------

st.header("Mean Similarity per Fold")

fig, ax = plt.subplots()

means = [filtered[filtered["fold"] == fold]["sim"].mean() for fold in selected_folds]

ax.bar(selected_folds, means)
ax.set_xlabel("Fold")
ax.set_ylabel("Mean Similarity")
ax.set_ylim(0, 1)

st.pyplot(fig)


# -----------------------------------------------------------

st.header("Similarity Distribution")

fig, ax = plt.subplots()

ax.hist(filtered["sim"], bins=30, edgecolor="black")
ax.set_xlabel("Cosine Similarity")
ax.set_ylabel("Count")

st.pyplot(fig)


# -----------------------------------------------------------

st.header("Split Comparison Across Folds")

metric = st.selectbox("Metric", ["mean", "median", "std", "min", "max"])

fig, ax = plt.subplots()

x = range(len(selected_folds))
width = 0.25

for i, split in enumerate(all_splits):
    split_df = df[(df["fold"].isin(selected_folds)) & (df["split"] == split)]
    values = [getattr(split_df[split_df["fold"] == fold]["sim"], metric)() for fold in selected_folds]
    ax.bar([xi + i * width for xi in x], values, width, label=split)

ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(selected_folds)
ax.set_ylabel(metric.capitalize())
ax.set_xlabel("Fold")
ax.legend()

st.pyplot(fig)


# -----------------------------------------------------------

st.header("Raw Similarity Values")

st.dataframe(filtered[["fold", "split", "task_id", "impl_index", "sim"]], use_container_width=True)


# -----------------------------------------------------------

csv = filtered.to_csv(index=False)

st.download_button("Download CSV", csv, "similarities_filtered.csv", "text/csv")