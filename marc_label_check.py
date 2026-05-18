from datasets import load_dataset

ds = load_dataset("shunk031/JGLUE", "MARC-ja", trust_remote_code=True)

print("=== Label info ===")
print(ds["train"].features)
print(ds["train"].features["label"])

from collections import Counter
print("\n=== Train label distribution ===")
print(Counter(ds["train"]["label"]))

print("\n=== Sample by label ===")
for label_id in [0, 1]:
    sample = next(x for x in ds["train"] if x["label"] == label_id)
    print(f"\nLabel {label_id}: {sample['sentence'][:100]}...")
    