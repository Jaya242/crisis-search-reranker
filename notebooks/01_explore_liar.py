"""
Day 1 Block 2: Explore the LIAR dataset.
Loads LIAR from local TSV files (downloaded from UCSB source).
Goal: understand structure, label distribution, sample statements.
"""

import random
from collections import Counter
import pandas as pd

# LIAR TSV column names (from the dataset's README)
COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title",
    "state", "party", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts",
    "context",
]

print("Loading LIAR from local TSV files...")
train = pd.read_csv("data/liar/train.tsv", sep="\t", names=COLUMNS)
valid = pd.read_csv("data/liar/valid.tsv", sep="\t", names=COLUMNS)
test = pd.read_csv("data/liar/test.tsv", sep="\t", names=COLUMNS)

print("\n=== Dataset structure ===")
print(f"  Train: {len(train)} rows")
print(f"  Valid: {len(valid)} rows")
print(f"  Test:  {len(test)} rows")

print("\n=== Columns ===")
print(list(train.columns))

print("\n=== First training sample ===")
sample = train.iloc[0]
for col in COLUMNS:
    print(f"  {col}: {sample[col]}")

print("\n=== Label distribution (train) ===")
counts = Counter(train["label"])
for label, count in sorted(counts.items()):
    print(f"  {label}: {count}")

print("\n=== 5 random statements per class ===")
random.seed(42)
for label in sorted(train["label"].unique()):
    subset = train[train["label"] == label]
    print(f"\n--- {label.upper()} ---")
    sample_rows = subset.sample(min(5, len(subset)), random_state=42)
    for stmt in sample_rows["statement"]:
        print(f"  - {stmt}")

print("\n=== Statement length stats (chars) ===")
lengths = train["statement"].str.len()
print(f"  Min: {lengths.min()}")
print(f"  Max: {lengths.max()}")
print(f"  Avg: {lengths.mean():.0f}")
print(f"  Median: {lengths.median():.0f}")

print("\n=== Done. ===")
