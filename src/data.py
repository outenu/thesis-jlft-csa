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
    """MARC-ja validation を読み込む"""
    ds = load_dataset("shunk031/JGLUE", "MARC-ja", trust_remote_code=True)
    return ds["validation"]

def load_wrime_binary():
    """WRIME test を reader 基準で2値化して読み込む（neutral 除外）"""
    ds = load_dataset("shunk031/wrime", "ver2", trust_remote_code=True)
    test = ds["test"]

    sentences = []
    labels = []
    for sample in test:
        sentiment = sample["avg_readers"]["sentiment"]
        if sentiment >= 1:
            sentences.append(sample["sentence"])
            labels.append(0)  # positive
        elif sentiment <= -1:
            sentences.append(sample["sentence"])
            labels.append(1)  # negative
        # sentiment == 0 は除外

    from datasets import Dataset
    return Dataset.from_dict({"sentence": sentences, "label": labels})