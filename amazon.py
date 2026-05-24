from datasets import load_dataset

ds = load_dataset("fancyzhx/amazon_polarity")
print(ds)
print(ds["train"][0])