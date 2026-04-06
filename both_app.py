import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

st.set_page_config(page_title="CodeBERT Analysis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("csv6_combined_flat.csv")
    df = df[df["sim"] >= 0.7]

    def minmax(s):
        return (s - s.min()) / (s.max() - s.min())

    df["sim_scaled"]   = minmax(df["sim"])         
    df["dist_scaled"]  = minmax(df["distance"])  
    df["dist_norm_scaled"]  = minmax(df["distance_norm"])  
    return df

df = load_data()


def get_metrics(scale):
    if scale:
        return ["sim_scaled", "dist_norm_scaled", "dist_scaled"]
    else:
        return ["sim", "distance_norm", "distance"]

st.title("CodeBERT Semantic Similarity Analysis")

scale = st.checkbox("Use min-max scaling for similarity and distance normalized", value=False)

# --- Controls ---
all_folds  = sorted(df["fold"].unique())
all_splits = ["fit", "validate", "test"]

col1, col2, col3 = st.columns(3)

with col1:
    selected_fold = st.selectbox("Fold", all_folds)

with col2:
    selected_split = st.selectbox("Split", all_splits)

with col3:
    n_rows = st.slider("Number of rows", 10, 1200, 50, key = "df_rows")

# --- Filter ---
filtered = df[(df["fold"] == selected_fold) & (df["split"] == selected_split)].head(n_rows)

st.write(f"Showing {len(filtered)} of {len(df[(df['fold'] == selected_fold) & (df['split'] == selected_split)])} records")

# --- Table ---
st.dataframe(filtered, use_container_width=True)

#--------------------------------------#
#--------------------------------------#

# --- Distribution ---
st.header("Distribution")

col1, col2 = st.columns(2)

with col1:
    dist_folds = st.multiselect("Folds", all_folds, default=all_folds, key="dist_folds")

with col2:
    dist_splits = st.multiselect("Splits", all_splits, default=all_splits, key="dist_splits")

dist_filtered = df[
    (df["fold"].isin(dist_folds)) &
    (df["split"].isin(dist_splits))
]

bins = st.slider("Bins", 10, 100, 30, key = "dist_bins")

metrics = get_metrics(scale)

fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))

for i, metric in enumerate(metrics):
    axes[i].hist(dist_filtered[metric], bins=bins, edgecolor="black")
    axes[i].set_title(metric)
    axes[i].set_xlabel(metric)
    axes[i].set_ylabel("Count")

plt.tight_layout()
st.pyplot(fig)
#--------------------------------------#
#--------------------------------------#

# --- Stats Table ---
st.header("Statistics per Fold and Split")

stats_split = st.selectbox("Split", all_splits, index=all_splits.index("test"), key="stats_split")

for metric in get_metrics(scale):
    st.subheader(f"{metric} Statistics")
    rows = []
    for fold in all_folds:
        subset = df[(df["fold"] == fold) & (df["split"] == stats_split)][metric]
        rows.append({
            "fold":   fold,
            "mean":   round(subset.mean(),   4),
            "median": round(subset.median(), 4),
            "min":    round(subset.min(),    4),
            "max":    round(subset.max(),    4),
            "var":    round(subset.var(),    4),
            "std":    round(subset.std(),    4),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

#--------------------------------------#
#--------------------------------------#

# --- Line Graph ---
st.header("Metric Trends Across Folds")

col1, col2 = st.columns(2)

with col1:
    line_metric = st.selectbox("Metric", get_metrics(scale), key="line_metric")

with col2:
    line_split = st.selectbox("Split", all_splits, index=all_splits.index("test"), key="line_split")

line_data = []

for fold in all_folds:
    subset = df[(df["fold"] == fold) & (df["split"] == line_split)][line_metric]
    line_data.append({
        "fold":   fold,
        "mean":   subset.mean(),
        "median": subset.median(),
        "min":    subset.min(),
        "max":    subset.max(),
    })

line_df = pd.DataFrame(line_data)

fig, ax = plt.subplots(figsize=(10, 5))

for col in ["mean", "median", "min", "max"]:
    ax.plot(line_df["fold"], line_df[col], marker="o", label=col)

ax.set_xlabel("Fold")
ax.set_ylabel(line_metric)
ax.set_title(f"{line_metric} across folds ({line_split})")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

#--------------------------------------#
#--------------------------------------#

# --- Correlation ---
st.header("Correlation Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    corr_folds = st.multiselect("Folds", all_folds, default=all_folds, key="corr_folds")

with col2:
    corr_splits = st.multiselect("Splits", all_splits, default=all_splits, key="corr_splits")

with col3:
    if scale:
        x_metric = st.selectbox("X axis", ["sim_scaled", "dist_scaled", "dist_norm_scaled"], index = 0, key="corr_x")
        y_metric = st.selectbox("Y axis", ["sim_scaled", "dist_scaled", "dist_norm_scaled"], index = 2, key="corr_y")
    else:
        x_metric = st.selectbox("X axis", ["sim", "distance", "distance_norm"], index = 0, key="corr_x")
        y_metric = st.selectbox("Y axis", ["sim", "distance", "distance_norm"], index = 2, key="corr_y")

corr_filtered = df[
    (df["fold"].isin(corr_folds)) &
    (df["split"].isin(corr_splits))
]

# --- Correlation numbers ---
pearson_r,  pearson_p  = scipy_stats.pearsonr( corr_filtered[x_metric], corr_filtered[y_metric])
spearman_r, spearman_p = scipy_stats.spearmanr(corr_filtered[x_metric], corr_filtered[y_metric])
kendall_r,  kendall_p  = scipy_stats.kendalltau(corr_filtered[x_metric], corr_filtered[y_metric])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Pearson r",  f"{pearson_r:.4f}",  f"p = {pearson_p:.2e}")

with col2:
    st.metric("Spearman r", f"{spearman_r:.4f}", f"p = {spearman_p:.2e}")

with col3:
    st.metric("Kendall τ",  f"{kendall_r:.4f}",  f"p = {kendall_p:.2e}")

# --- Scatter plot ---
sample_size = st.slider("Sample size", 100, len(corr_filtered), min(2000, len(corr_filtered)), key="corr_sample")

sample = corr_filtered.sample(n=sample_size, random_state=42)

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(sample[x_metric], sample[y_metric], alpha=0.3, s=10)
ax.set_xlabel(x_metric)
ax.set_ylabel(y_metric)
ax.set_title(f"{x_metric} vs {y_metric}  |  Pearson: {pearson_r:.4f}  Spearman: {spearman_r:.4f}")

plt.tight_layout()
st.pyplot(fig)