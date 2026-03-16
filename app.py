# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lev_analysis import process_folds


# -----------------------------------------------------------
# Page Title
# -----------------------------------------------------------

st.title("Levenshtein Distance Analysis")

st.write(
"""
This dashboard analyzes the syntactic distance between **correct and incorrect implementations**
using Levenshtein distance.

You can explore:
- statistics per fold
- how folds vary from each other
- how splits (fit / validate / test) compare
- pairwise differences between splits
"""
)


# -----------------------------------------------------------
# Controls
# -----------------------------------------------------------

st.header("Controls")

normalize = st.checkbox(
    "Normalize Distance",
    help="Normalize distance by length of the correct implementation"
)

# compute statistics for all folds
results = process_folds(normalize)

all_folds = list(results.keys())

# choose which folds to analyze
selected_folds = st.multiselect(
    "Select folds to analyze",
    all_folds,
    default=all_folds
)

# choose split
split = st.selectbox(
    "Select split",
    ["fit", "validate", "test"]
)


# -----------------------------------------------------------
# SECTION 1 — Statistics Table
# -----------------------------------------------------------

st.header("Statistics Table")

st.write(
"""
Shows summary statistics of Levenshtein distances for each fold.

Rows = folds  
Columns = statistics (Mean, Median, Std, Min, Max)

Each value summarizes the distribution of distances between the
correct implementation and incorrect implementations.
"""
)

rows = []

for fold in selected_folds:
    stats = results[fold][split]["stats"]

    row = {"Fold": fold}
    row.update(stats)

    rows.append(row)

df = pd.DataFrame(rows).set_index("Fold")

st.dataframe(df)


# -----------------------------------------------------------
# SECTION 2 — Fold Comparison Plot
# -----------------------------------------------------------

st.header("Fold Comparison Plot")

st.write(
"""
This plot shows how statistical metrics change **across folds**.

Each line represents a statistic:
- Mean distance
- Median distance
- Standard deviation
- Minimum distance
- Maximum distance
"""
)

metrics = ["Mean", "Median", "Std", "Min", "Max"]

fig, ax = plt.subplots()

for metric in metrics:

    values = [
        results[fold][split]["stats"][metric]
        for fold in selected_folds
    ]

    ax.plot(selected_folds, values, marker="o", label=metric)

ax.set_xlabel("Fold")
ax.set_ylabel("Distance Value")
ax.legend()

st.pyplot(fig)


# -----------------------------------------------------------
# SECTION 3 — Split Comparison (Grouped Bar Chart)
# -----------------------------------------------------------

st.header("Split Comparison Across Folds")

st.write(
"""
This grouped bar chart compares **fit, validate, and test splits**
for each fold using a selected metric.
"""
)

metric_bar = st.selectbox(
    "Metric for split comparison",
    ["Mean", "Median", "Min", "Max", "Std"]
)

folds = selected_folds
splits = ["fit", "validate", "test"]

x = np.arange(len(folds))
width = 0.25

fig, ax = plt.subplots()

for i, split_name in enumerate(splits):

    values = [
        results[fold][split_name]["stats"][metric_bar]
        for fold in folds
    ]

    ax.bar(
        x + i * width,
        values,
        width,
        label=split_name
    )

ax.set_xticks(x + width)
ax.set_xticklabels(folds)

ax.set_ylabel(metric_bar)
ax.set_xlabel("Fold")
ax.legend()

st.pyplot(fig)


# -----------------------------------------------------------
# SECTION 4 — Pairwise Split Differences
# -----------------------------------------------------------

st.header("Pairwise Split Differences")

st.write(
"""
This section compares two dataset splits directly.

For each fold, we compute the **absolute difference** between the
selected statistic of the two splits.

Example:

Mean distance (fit) = 210  
Mean distance (test) = 350  

Difference = |210 − 350| = 140

Large values indicate that the two splits behave differently,
while small values indicate similar distributions.
"""
)

# select which two splits to compare
pair = st.selectbox(
    "Select split comparison",
    [
        ("fit", "test"),
        ("fit", "validate"),
        ("validate", "test")
    ],
    format_func=lambda x: f"{x[0]} vs {x[1]}"
)

# select statistic to compare
metric_diff = st.selectbox(
    "Metric for difference",
    ["Mean", "Median", "Std"]
)

split_a, split_b = pair

# compute difference per fold
values = [
    abs(
        results[fold][split_a]["stats"][metric_diff] -
        results[fold][split_b]["stats"][metric_diff]
    )
    for fold in selected_folds
]

# -----------------------------------------------------------
# Plot
# -----------------------------------------------------------

fig, ax = plt.subplots()

ax.bar(selected_folds, values)

ax.set_xlabel("Fold")
ax.set_ylabel(f"{metric_diff} Difference")
ax.set_title(f"{split_a} vs {split_b}")

st.pyplot(fig)


# -----------------------------------------------------------
# Table of values
# -----------------------------------------------------------

df_diff = pd.DataFrame({
    "Fold": selected_folds,
    "Difference": values
})

st.dataframe(df_diff.set_index("Fold"))