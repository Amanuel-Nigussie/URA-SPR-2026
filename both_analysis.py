import os
import csv
import numpy as np
import json
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.distance import Levenshtein


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
def compute_distances(tasks):

    distances = {}

    for task in tasks:

        implementations = task["implementations"]
        correct = implementations[0]

        distances[task["task_id"]] = []

        for impl in implementations[1:]:

            dist      = Levenshtein.distance(correct, impl)
            dist_norm = dist / len(correct) if len(correct) > 0 else 0

            distances[task["task_id"]].append((dist, dist_norm))

    return distances


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
def save_csvs(results):

    # --- CSV 1: sims as list, one row per task ---
    with open("csv1_sims_list.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "split", "task_id", "sims"])
        for fold, splits in results.items():
            for split, data in splits.items():
                for task_id, sims in data["similarities"].items():
                    writer.writerow([fold, split, task_id, sims])
    print("Saved csv1_sims_list.csv")

    # --- CSV 2: sim per implementation, flat ---
    with open("csv2_sims_flat.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "split", "task_id", "impl_index", "sim"])
        for fold, splits in results.items():
            for split, data in splits.items():
                for task_id, sims in data["similarities"].items():
                    for impl_idx, sim in enumerate(sims, start=0):
                        writer.writerow([fold, split, task_id, impl_idx, sim])
    print("Saved csv2_sims_flat.csv")

    # --- CSV 3: distances as list, one row per task ---
    with open("csv3_distances_list.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "split", "task_id", "distances", "distances_norm"])
        for fold, splits in results.items():
            for split, data in splits.items():
                for task_id, dist_pairs in data["distances"].items():
                    dists      = [d  for d, dn in dist_pairs]
                    dists_norm = [dn for d, dn in dist_pairs]
                    writer.writerow([fold, split, task_id, dists, dists_norm])
    print("Saved csv3_distances_list.csv")

    # --- CSV 4: distance per implementation, flat ---
    with open("csv4_distances_flat.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "split", "task_id", "impl_index", "distance", "distance_norm"])
        for fold, splits in results.items():
            for split, data in splits.items():
                for task_id, dist_pairs in data["distances"].items():
                    for impl_idx, (dist, dist_norm) in enumerate(dist_pairs, start=0):
                        writer.writerow([fold, split, task_id, impl_idx, dist, dist_norm])
    print("Saved csv4_distances_flat.csv")

    # --- CSV 5: sims and distances as lists, one row per task ---
    with open("csv5_combined_list.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "split", "task_id", "sims", "distances", "distances_norm"])
        for fold, splits in results.items():
            for split, data in splits.items():
                for task_id, sims in data["similarities"].items():
                    dist_pairs = data["distances"][task_id]
                    dists      = [d  for d, dn in dist_pairs]
                    dists_norm = [dn for d, dn in dist_pairs]
                    writer.writerow([fold, split, task_id, sims, dists, dists_norm])
    print("Saved csv5_combined_list.csv")

    # --- CSV 6: sim and distance per implementation, flat ---
    with open("csv6_combined_flat.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "split", "task_id", "impl_index", "sim", "distance", "distance_norm"])
        for fold, splits in results.items():
            for split, data in splits.items():
                for task_id, sims in data["similarities"].items():
                    dist_pairs = data["distances"][task_id]
                    for impl_idx, (sim, (dist, dist_norm)) in enumerate(zip(sims, dist_pairs), start=0):
                        writer.writerow([fold, split, task_id, impl_idx, sim, dist, dist_norm])
    print("Saved csv6_combined_flat.csv")


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
            distances    = compute_distances(tasks)

            sim_values  = [sim  for sims       in similarities.values() for sim     in sims]
            dist_values = [dist for dist_pairs in distances.values()    for dist, _ in dist_pairs]

            results[f"fold_{i}"][split] = {
                "similarities": similarities,
                "distances":    distances,
                "stats_sim":    compute_stats(sim_values),
                "stats_dist":   compute_stats(dist_values),
            }

    print("\nFinished processing all folds.")

    save_csvs(results)

    return results


# -----------------------------

if __name__ == "__main__":
    process_folds()