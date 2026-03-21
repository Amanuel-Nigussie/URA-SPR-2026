import os
import csv
import numpy as np
import json
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


DATASET_PATH = os.path.join("x_fold", "bcb")


# -----------------------------
def load_jsonl(file_path):

    tasks = []

    with open(file_path, "r") as f:
        for line in f:
            tasks.append(json.loads(line))

    return tasks


# -----------------------------
def load_model():

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    model.eval()

    return tokenizer, model


# -----------------------------
def get_embedding(code, tokenizer, model):

    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.numpy()


# -----------------------------
def compute_similarities(tasks, tokenizer, model, embedding_cache):

    similarities = {}

    total_tasks = len(tasks)

    def get_cached_embedding(code):
        if code not in embedding_cache:
            embedding_cache[code] = get_embedding(code, tokenizer, model)
            print("[COMPUTED]", end=" ")
        else:
            print("[CACHED]", end=" ")
        return embedding_cache[code]

    for task_idx, task in enumerate(tasks, start=1):

        print(f"\n  Task {task_idx}/{total_tasks}", end=" ")

        implementations = task["implementations"]
        correct = implementations[0]

        correct_embedding = get_cached_embedding(correct)
        similarities[task["task_id"]] = []

        for impl in implementations[1:]:

            incorrect_embedding = get_cached_embedding(impl)

            sim = cosine_similarity(correct_embedding, incorrect_embedding)[0][0]

            similarities[task["task_id"]].append(sim)

    return similarities


# -----------------------------
def compute_stats(values):

    if len(values) == 0:
        return {}

    return {
        "Count":  len(values),
        "Mean":   float(np.mean(values)),
        "Median": float(np.median(values)),
        "Std":    float(np.std(values)),
        "Min":    float(np.min(values)),
        "Max":    float(np.max(values)),
    }


# -----------------------------
def save_csv(results, output_path="similarities.csv", output_path1="similarities1.csv"):

    with open(output_path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["fold", "split", "task_id", "impl_index", "sim"])

        for fold, splits in results.items():
            for split, data in splits.items():
                for task_id, sims in data["similarities"].items():
                    for impl_idx, sim in enumerate(sims, start=1):
                        writer.writerow([fold, split, task_id, impl_idx, sim])

    print(f"\nSaved to {output_path}")

    with open(output_path1, "w", newline="") as f1:

        writer1 = csv.writer(f1)
        writer1.writerow(["fold", "split", "task_id", "sims"])

        for fold, splits in results.items():
            for split, data in splits.items():
                for task_id, sims in data["similarities"].items():
                    writer1.writerow([fold, split, task_id, sims])

    print(f"\nSaved to {output_path1}")


# -----------------------------
def process_folds():

    tokenizer, model = load_model()
    embedding_cache = {}

    results = {}

    for i in range(10):

        print(f"\nProcessing fold {i}")

        results[f"fold_{i}"] = {}

        for split in ["fit", "validate", "test"]:

            print(f"\n  Split: {split}")

            file_path = os.path.join(DATASET_PATH, f"fold_{i}", f"{split}.jsonl")

            tasks = load_jsonl(file_path)

            similarities = compute_similarities(tasks, tokenizer, model, embedding_cache)

            stats = compute_stats([sim for task_id in similarities for sim in similarities[task_id]])

            results[f"fold_{i}"][split] = {"similarities": similarities, "stats": stats}

    print("\nFinished processing all folds.")

    save_csv(results)

    return results

# -----------------------------

if __name__ == "__main__":
    process_folds()
