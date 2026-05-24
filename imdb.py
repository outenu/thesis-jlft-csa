from datasets import load_dataset
ds = load_dataset("imdb")
print(ds)
print(ds["train"][0])
