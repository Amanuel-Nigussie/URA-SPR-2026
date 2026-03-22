import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

st.set_page_config(page_title="CodeBERT Analysis", layout="wide")


# -----------------------------------------------------------
# Load data
# -----------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("csv6_combined_flat.csv")


df = load_data()

all_folds  = sorted(df["fold"].unique())
all_splits = ["fit", "validate", "test"]


# -----------------------------------------------------------
# Sidebar — global controls
# -----------------------------------------------------------

st.sidebar.title("Controls")

selected_folds = st.sidebar.multiselect("Folds", all_folds, default=all_folds)
selected_splits = st.sidebar.multiselect("Splits", all_splits, default=all_splits)

filtered = df[
    (df["fold"].isin(selected_folds)) &
    (df["split"].isin(selected_splits))
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Records:** {len(filtered):,}")
st.sidebar.markdown(f"**Tasks:** {filtered['task_id'].nunique():,}")


# -----------------------------------------------------------
# Title
# -----------------------------------------------------------

st.title("CodeBERT Semantic Similarity Analysis")
st.write("Analyzing cosine similarity and Levenshtein distance between correct and incorrect implementations.")


# -----------------------------------------------------------
# 1. Distribution Analysis
# -----------------------------------------------------------

st.header("1. Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    bins_sim = st.slider("Bins (similarity)", 10, 100, 30, key="bins_sim")

with col2:
    bins_dist = st.slider("Bins (distance)", 10, 100, 30, key="bins_dist")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(filtered["sim"], bins=bins_sim, edgecolor="black")
axes[0].set_title("Cosine Similarity")
axes[0].set_xlabel("Similarity")
axes[0].set_ylabel("Count")

axes[1].hist(filtered["distance"], bins=bins_dist, edgecolor="black")
axes[1].set_title("Levenshtein Distance (raw)")
axes[1].set_xlabel("Distance")
axes[1].set_ylabel("Count")

axes[2].hist(filtered["distance_norm"], bins=bins_dist, edgecolor="black")
axes[2].set_title("Levenshtein Distance (normalized)")
axes[2].set_xlabel("Normalized Distance")
axes[2].set_ylabel("Count")

plt.tight_layout()
st.pyplot(fig)


# -----------------------------------------------------------
# 2. Similarity vs Distance Scatter
# -----------------------------------------------------------

st.header("2. Similarity vs Distance")

st.write("""
Each point is one incorrect implementation. Quadrants tell you:
- **Top-left** (high sim, high dist): different code structure, similar meaning
- **Bottom-left** (low sim, high dist): completely different
- **Bottom-right** (low sim, low dist): looks similar but behaves differently — subtle bug
- **Top-right** (high sim, low dist): minor style/formatting difference
""")

col1, col2 = st.columns(2)

with col1:
    dist_type = st.radio("Distance type", ["distance", "distance_norm"], horizontal=True)

with col2:
    sample_size = st.slider("Sample size (for readability)", 100, len(filtered), min(2000, len(filtered)), step=100)

sample = filtered.sample(n=sample_size, random_state=42) if len(filtered) > sample_size else filtered

fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(
    sample["sim"],
    sample[dist_type],
    alpha=0.3,
    s=10,
    c=sample["sim"],
    cmap="coolwarm"
)

plt.colorbar(scatter, ax=ax, label="Cosine Similarity")
ax.set_xlabel("Cosine Similarity")
ax.set_ylabel(dist_type)
ax.set_title("Cosine Similarity vs Levenshtein Distance")

st.pyplot(fig)


# -----------------------------------------------------------
# 3. Fold Consistency
# -----------------------------------------------------------

st.header("3. Fold Consistency")

metric_fold = st.selectbox("Metric", ["sim", "distance", "distance_norm"], key="metric_fold")

fold_stats = filtered.groupby("fold")[metric_fold].agg(["mean", "std", "median"]).reindex(selected_folds)

fig, ax = plt.subplots(figsize=(10, 4))

x = np.arange(len(selected_folds))
ax.bar(x, fold_stats["mean"], yerr=fold_stats["std"], capsize=4, edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(selected_folds, rotation=45)
ax.set_ylabel(f"Mean {metric_fold}")
ax.set_title(f"Mean {metric_fold} per Fold (error bars = std)")

st.pyplot(fig)
st.dataframe(fold_stats.round(4), use_container_width=True)


# -----------------------------------------------------------
# 4. Split Comparison
# -----------------------------------------------------------

st.header("4. Split Comparison Across Folds")

metric_split = st.selectbox("Metric", ["sim", "distance", "distance_norm"], key="metric_split")

x = np.arange(len(selected_folds))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))

for i, split in enumerate(selected_splits):
    split_df = df[(df["fold"].isin(selected_folds)) & (df["split"] == split)]
    values = [split_df[split_df["fold"] == fold][metric_split].mean() for fold in selected_folds]
    ax.bar(x + i * width, values, width, label=split, edgecolor="black")

ax.set_xticks(x + width)
ax.set_xticklabels(selected_folds, rotation=45)
ax.set_ylabel(f"Mean {metric_split}")
ax.set_xlabel("Fold")
ax.legend()
ax.set_title(f"Mean {metric_split} per Fold and Split")

st.pyplot(fig)


# -----------------------------------------------------------
# 5. Correlation
# -----------------------------------------------------------

st.header("5. Correlation Between Metrics")

col1, col2 = st.columns(2)

with col1:
    pearson_r, pearson_p = scipy_stats.pearsonr(filtered["sim"], filtered["distance_norm"])
    st.metric("Pearson Correlation (sim vs dist_norm)", f"{pearson_r:.4f}", f"p = {pearson_p:.2e}")

with col2:
    spearman_r, spearman_p = scipy_stats.spearmanr(filtered["sim"], filtered["distance_norm"])
    st.metric("Spearman Correlation (sim vs dist_norm)", f"{spearman_r:.4f}", f"p = {spearman_p:.2e}")

st.write("""
- **Pearson**: linear correlation
- **Spearman**: rank-based correlation, more robust to outliers
- Values close to -1 mean the metrics strongly agree (high similarity = low distance)
- Values close to 0 mean they capture different things
""")


# -----------------------------------------------------------
# 6. Per impl_index Analysis
# -----------------------------------------------------------

st.header("6. Per Implementation Index Analysis")

metric_impl = st.selectbox("Metric", ["sim", "distance", "distance_norm"], key="metric_impl")

impl_stats = filtered.groupby("impl_index")[metric_impl].agg(["mean", "std", "count"]).reset_index()

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(impl_stats["impl_index"], impl_stats["mean"], yerr=impl_stats["std"], capsize=4, edgecolor="black")
ax.set_xlabel("Implementation Index")
ax.set_ylabel(f"Mean {metric_impl}")
ax.set_title(f"Mean {metric_impl} by Implementation Index")
ax.set_xticks(impl_stats["impl_index"])

st.pyplot(fig)
st.dataframe(impl_stats.round(4), use_container_width=True)


# -----------------------------------------------------------
# 7. Outlier Analysis
# -----------------------------------------------------------

st.header("7. Outlier Analysis")

col1, col2 = st.columns(2)

with col1:
    n_outliers = st.slider("Number of outliers to show", 5, 50, 10)

with col2:
    outlier_metric = st.selectbox("Outlier metric", ["sim", "distance_norm"])

st.subheader(f"Lowest {outlier_metric} (most different from correct)")
st.dataframe(
    filtered.nsmallest(n_outliers, outlier_metric)[["fold", "split", "task_id", "impl_index", "sim", "distance", "distance_norm"]],
    use_container_width=True
)

st.subheader(f"Highest {outlier_metric} (most similar to correct)")
st.dataframe(
    filtered.nlargest(n_outliers, outlier_metric)[["fold", "split", "task_id", "impl_index", "sim", "distance", "distance_norm"]],
    use_container_width=True
)


# -----------------------------------------------------------
# Raw data + download
# -----------------------------------------------------------

st.header("Raw Data")

st.dataframe(filtered, use_container_width=True)

st.download_button(
    "Download filtered CSV",
    filtered.to_csv(index=False),
    "filtered.csv",
    "text/csv"
)               