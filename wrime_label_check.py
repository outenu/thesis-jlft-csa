from datasets import load_dataset
from collections import Counter

ds = load_dataset("shunk031/wrime", "ver2", trust_remote_code=True)

print("=== Writer sentiment distribution (train) ===")
writer_sentiments = [x["writer"]["sentiment"] for x in ds["train"]]
print(Counter(writer_sentiments))

print("\n=== Avg readers sentiment distribution (train) ===")
reader_sentiments = [x["avg_readers"]["sentiment"] for x in ds["train"]]
print(Counter(reader_sentiments))

print("\n=== Binary (readers, neutral excluded) ===")
binary = []
for s in reader_sentiments:
    if s >= 1: binary.append("positive")
    elif s <= -1: binary.append("negative")
print(Counter(binary))