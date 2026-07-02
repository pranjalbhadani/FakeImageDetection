from collections import Counter

from src.data.dataset import load_dataset
from src.split import create_splits

paths, labels = load_dataset("dataset")

(
    train_paths,
    train_labels,
    val_paths,
    val_labels,
    test_paths,
    test_labels,
) = create_splits(paths, labels)

print("TRAIN")
print(len(train_paths), Counter(train_labels))

print()

print("VALIDATION")
print(len(val_paths), Counter(val_labels))

print()

print("TEST")
print(len(test_paths), Counter(test_labels))