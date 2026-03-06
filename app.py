# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from analysis import process_folds


st.title("Fold Analysis")


normalize = st.checkbox("Normalize Distance")

results = process_folds(normalize)

all_folds = list(results.keys())

selected_folds = st.multiselect("Select Folds", all_folds,default=all_folds)

split = st.selectbox(
    "Select Split",
    ["fit", "validate", "test"]
)



rows = []

for fold in selected_folds:
    stats = results[fold][split]["stats"]
    row = {"Fold": fold}
    row.update(stats)
    rows.append(row)

df = pd.DataFrame(rows)
df = df.set_index("Fold")

st.subheader("Statistics Table")
st.dataframe(df)


st.subheader("Fold Comparison Plot")

metrics = ["Mean", "Std", "Min", "Max"]

fig, ax = plt.subplots()

for metric in metrics:
    values = []
    for fold in selected_folds:
        values.append(results[fold][split]["stats"][metric])
    ax.plot(selected_folds, values, marker="o", label=metric)

ax.set_xlabel("Fold")
ax.set_ylabel("Value")
ax.legend()

st.pyplot(fig)