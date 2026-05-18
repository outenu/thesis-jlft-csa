from datasets import load_dataset
ds = load_dataset("shunk031/JGLUE", "MARC-ja")
print(ds)
print(ds["train"][0])