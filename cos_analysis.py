import os
import numpy as np
import json
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


DATASET_PATH = os.path.join("x_fold", "bcb")


# -----------------------------
# Load dataset
# -----------------------------
def load_jsonl(file_path):

    tasks = []

    with open(file_path, "r") as f:
        for line in f:
            tasks.append(json.loads(line))

    return tasks


# -----------------------------
# Load CodeBERT model
# -----------------------------
def load_model():

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    model.eval()

    return tokenizer, model


# -----------------------------
# Compute embedding
# -----------------------------
def get_embedding(code, tokenizer, model):

    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.numpy()


# -----------------------------
# Compute cosine similarities
# -----------------------------
def compute_similarities(tasks, tokenizer, model):

    similarities = []
    embedding_cache = {}

    def get_cached_embedding(code):

        if code not in embedding_cache:
            embedding_cache[code] = get_embedding(code, tokenizer, model)

        return embedding_cache[code]

    for task in tasks:

        implementations = task["implementations"]
        correct = implementations[0]

        correct_embedding = get_cached_embedding(correct)

        for impl in implementations[1:]:

            incorrect_embedding = get_cached_embedding(impl)

            sim = cosine_similarity(correct_embedding, incorrect_embedding)[0][0]

            similarities.append(sim)

    return similarities


# -----------------------------
# Compute statistics
# -----------------------------
def compute_stats(values):

    if len(values) == 0:
        return {}

    return {"Count": len(values), "Mean": float(np.mean(values)), "Median": float(np.median(values)), "Std": float(np.std(values)), "Min": float(np.min(values)), "Max": float(np.max(values))}


# -----------------------------
# Process all folds
# -----------------------------
def process_folds():

    tokenizer, model = load_model()

    results = {}

    for i in range(10):

        results[f"fold_{i}"] = {}

        for split in ["fit", "validate", "test"]:

            file_path = os.path.join(DATASET_PATH, f"fold_{i}", f"{split}.jsonl")

            tasks = load_jsonl(file_path)

            similarities = compute_similarities(tasks, tokenizer, model)

            stats = compute_stats(similarities)

            results[f"fold_{i}"][split] = {"similarities": similarities, "stats": stats}

    return results