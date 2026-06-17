from datasets import load_dataset

def load_imdb(n_samples=None):
    """IMDB を読み込む。n_samples 指定時はサブセット"""
    if n_samples:
        ds = load_dataset("imdb", split=f"train[:{n_samples}]")
    else:
        ds = load_dataset("imdb", split="train")
    test = load_dataset("imdb", split="test")
    return ds, test

def load_marc_ja():
    ds = load_dataset("shunk031/JGLUE", "MARC-ja", trust_remote_code=True)
    val = ds["validation"]
    # MARC-ja(0=pos,1=neg) → 標準(0=neg,1=pos) に反転
    def flip_label(example):
        example["label"] = 1 - example["label"]
        return example
    return val.map(flip_label)

def load_wrime_binary():
    ds = load_dataset("shunk031/wrime", "ver2", trust_remote_code=True)
    test = ds["test"]
    sentences, labels = [], []
    for sample in test:
        sentiment = sample["avg_readers"]["sentiment"]
        if sentiment >= 1:
            sentences.append(sample["sentence"])
            labels.append(1)  # positive（標準に合わせる）
        elif sentiment <= -1:
            sentences.append(sample["sentence"])
            labels.append(0)  # negative
    from datasets import Dataset
    return Dataset.from_dict({"sentence": sentences, "label": labels})