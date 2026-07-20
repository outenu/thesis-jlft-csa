from datasets import load_dataset, Dataset, concatenate_datasets

def load_imdb(n_train=None):
    """IMDB（negative=0, positive=1、標準のまま）"""
    split = f"train[:{n_train}]" if n_train else "train"
    train = load_dataset("imdb", split=split).shuffle(seed=42)
    test = load_dataset("imdb", split="test")
    train = train.rename_column("text", "sentence")
    test = test.rename_column("text", "sentence")
    return train, test


def load_amazon(n_train=25000):
    """Amazon Polarity（negative=0, positive=1、標準のまま）"""
    train = load_dataset("fancyzhx/amazon_polarity", split=f"train[:{n_train * 3}]")
    train = train.shuffle(seed=42).select(range(n_train))
    train = train.rename_column("content", "sentence")
    return train


def _flip(example):
    example["label"] = 1 - example["label"]
    return example


def load_marc_ja_val():
    """MARC-ja validation（0=pos,1=neg → 反転して 0=neg,1=pos）"""
    ds = load_dataset("shunk031/JGLUE", "MARC-ja", trust_remote_code=True)
    return ds["validation"].map(_flip)


def load_marc_ja_train(n_train=25000, balanced=True):
    """MARC-ja train（単言語モデル学習用、均衡サンプリング）"""
    ds = load_dataset("shunk031/JGLUE", "MARC-ja", trust_remote_code=True)["train"]
    ds = ds.map(_flip) 
    if balanced:
        neg = ds.filter(lambda x: x["label"] == 0).shuffle(seed=42)
        pos = ds.filter(lambda x: x["label"] == 1).shuffle(seed=42)
        n = min(len(neg), len(pos), n_train // 2)
        ds = concatenate_datasets([neg.select(range(n)), pos.select(range(n))])
        ds = ds.shuffle(seed=42)
    else:
        ds = ds.shuffle(seed=42).select(range(min(n_train, len(ds))))
    return ds


def load_wrime_binary():
    """WRIME test（reader基準2値化、negative=0, positive=1）"""
    test = load_dataset("shunk031/wrime", "ver2", trust_remote_code=True)["test"]
    sentences, labels = [], []
    for s in test:
        v = s["avg_readers"]["sentiment"]
        if v >= 1:
            sentences.append(s["sentence"])
            labels.append(1)
        elif v <= -1:
            sentences.append(s["sentence"])
            labels.append(0)
    return Dataset.from_dict({"sentence": sentences, "label": labels})
