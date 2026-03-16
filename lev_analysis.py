import os
import numpy as np
from rapidfuzz.distance import Levenshtein
import json

def load_jsonl(file_path):
    tasks = []
    with open(file_path, "r") as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


DATASET_PATH = os.path.join("x_fold", "bcb")


def compute_distances(tasks, normalize=False):
    distances = []

    for task in tasks:
        implementations = task["implementations"]
        correct = implementations[0]

        for impl in implementations[1:]:
            dist = Levenshtein.distance(correct, impl)

            if normalize:

                dist = dist / len(correct) if len(correct) > 0 else 0

            distances.append(dist)

    return distances


def compute_stats(distances):
    if len(distances) == 0:
        return {}

    return {
        "Count": len(distances),
        "Mean": float(np.mean(distances)),
        "Median": float(np.median(distances)),
        "Std": float(np.std(distances)),
        "Min": float(np.min(distances)),
        "Max": float(np.max(distances)),
    }


def process_folds(normalize=False):

    results = {}

    for i in range(10):

        results[f"fold_{i}"] = {}
        for split in ["fit", "validate", "test"]:
            file_path = os.path.join(DATASET_PATH, f"fold_{i}", f"{split}.jsonl")

            tasks = load_jsonl(file_path)
            distances = compute_distances(tasks, normalize)
            stats = compute_stats(distances)

            results[f"fold_{i}"][split] = {
                "distances": distances,
                "stats": stats,
            }

    return results