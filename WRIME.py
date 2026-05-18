from datasets import load_dataset
ds = load_dataset("shunk031/wrime", "ver2")
print(ds)
print(ds["train"][0])