import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cos_analysis import process_folds


# -----------------------------------------------------------

@st.cache_data
def compute_results():
    return process_folds()


# -----------------------------------------------------------

st.title("CodeBERT Semantic Similarity Analysis")

st.write("""
This dashboard analyzes the semantic similarity between **correct and incorrect implementations**
using **CodeBERT embeddings and cosine similarity**.

You can explore:
- statistics per fold
- how folds vary from each other
- how splits (fit / validate / test) compare
- pairwise differences between splits
""")


# -----------------------------------------------------------

st.header("Controls")

results = compute_results()

all_folds = list(results.keys())

selected_folds = st.multiselect("Select folds to analyze", all_folds, default=all_folds)

split = st.selectbox("Select split", ["fit", "validate", "test"])


# -----------------------------------------------------------

st.header("Statistics Table")

rows = []

for fold in selected_folds:
    stats = results[fold][split]["stats"]
    row = {"Fold": fold}
    row.update(stats)
    rows.append(row)

df_stats = pd.DataFrame(rows).set_index("Fold")

st.dataframe(df_stats)


# -----------------------------------------------------------

st.header("Fold Comparison Plot")

metrics = ["Mean", "Median", "Std", "Min", "Max"]

fig, ax = plt.subplots()

for metric in metrics:
    values = [results[fold][split]["stats"][metric] for fold in selected_folds]
    ax.plot(selected_folds, values, marker="o", label=metric)

ax.set_xlabel("Fold")
ax.set_ylabel("Similarity Value")
ax.legend()

st.pyplot(fig)


# -----------------------------------------------------------

st.header("Split Comparison Across Folds")

metric_bar = st.selectbox("Metric for split comparison", ["Mean", "Median", "Min", "Max", "Std"])

folds = selected_folds
splits = ["fit", "validate", "test"]

x = np.arange(len(folds))
width = 0.25

fig, ax = plt.subplots()

for i, split_name in enumerate(splits):
    values = [results[fold][split_name]["stats"][metric_bar] for fold in folds]
    ax.bar(x + i * width, values, width, label=split_name)

ax.set_xticks(x + width)
ax.set_xticklabels(folds)
ax.set_ylabel(metric_bar)
ax.set_xlabel("Fold")
ax.legend()

st.pyplot(fig)

# -----------------------------------------------------------

st.header("Pairwise Split Differences")

pair = st.selectbox("Select split comparison", [("fit","test"),("fit","validate"),("validate","test")], format_func=lambda x: f"{x[0]} vs {x[1]}")

metric_diff = st.selectbox("Metric for difference", ["Mean","Median","Std"])

split_a, split_b = pair

values = [abs(results[fold][split_a]["stats"][metric_diff] - results[fold][split_b]["stats"][metric_diff]) for fold in selected_folds]

fig, ax = plt.subplots()

ax.bar(selected_folds, values)

ax.set_xlabel("Fold")
ax.set_ylabel(f"{metric_diff} Difference")
ax.set_title(f"{split_a} vs {split_b}")

st.pyplot(fig)

df_diff = pd.DataFrame({"Fold": selected_folds, "Difference": values})

st.dataframe(df_diff.set_index("Fold"))


# -----------------------------------------------------------

st.header("Exact Similarity Values")

st.write("This table shows the raw cosine similarity scores between correct and incorrect implementations.")

all_rows = []

for fold in selected_folds:
    values = results[fold][split]["similarities"]
    for v in values:
        all_rows.append({"Fold": fold, "Split": split, "Similarity": v})

df_values = pd.DataFrame(all_rows)

st.dataframe(df_values)


# -----------------------------------------------------------

csv = df_values.to_csv(index=False)

st.download_button("Download Similarity CSV", csv, "codebert_similarity.csv", "text/csv")