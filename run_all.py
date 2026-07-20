import os
import json
import gc
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed
)
from src.data import (
    load_imdb, load_amazon, load_marc_ja_val, load_marc_ja_train, load_wrime_binary
)
from src.evaluate import evaluate

MAX_LEN = 256
EPOCHS = 3
BATCH = 32
EVAL_BATCH = 128
LR = 2e-5
N_TRAIN = 25000
SEEDS = [42, 123, 456]

MODELS = {
    "mBERT": "bert-base-multilingual-cased",
    "XLM-R": "xlm-roberta-base",
}
TOHOKU = "tohoku-nlp/bert-base-japanese-v3"


def tok_fn(tokenizer):
    def f(batch):
        return tokenizer(batch["sentence"], truncation=True,
                         padding="max_length", max_length=MAX_LEN)
    return f


def train_model(model_name, train_ds, seed, tag):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    train_ds = train_ds.map(tok_fn(tokenizer), batched=True)

    args = TrainingArguments(
        output_dir=f"/tmp/{tag}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    res = trainer.train()
    return model, tokenizer, res


@torch.no_grad()
def predict(model, tokenizer, dataset):
    model.eval()
    device = model.device
    preds, probs, labels = [], [], []
    for i in range(0, len(dataset), EVAL_BATCH):
        batch = dataset[i:i + EVAL_BATCH]
        enc = tokenizer(batch["sentence"], return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1)
        preds.extend(torch.argmax(p, dim=-1).cpu().tolist())
        probs.extend(p.max(dim=-1).values.cpu().tolist())
        labels.extend(batch["label"])
    return labels, preds, probs


def save_misclassified(dataset, y_true, y_pred, y_prob, path):
    rows = []
    for i, (t, p, c) in enumerate(zip(y_true, y_pred, y_prob)):
        if t != p:
            rows.append({
                "sentence": dataset[i]["sentence"],
                "true_label": int(t),
                "pred_label": int(p),
                "confidence": float(c),
            })
    rows.sort(key=lambda x: -x["confidence"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return len(rows)


def run_transfer(model_key, source, seed, marc_val, wrime):
    tag = f"{model_key}_{source}_seed{seed}"
    t0 = time.time()

    train_ds = load_imdb(N_TRAIN)[0] if source == "imdb" else load_amazon(N_TRAIN)
    model, tokenizer, res = train_model(MODELS[model_key], train_ds, seed, tag)

    out = {
        "tag": tag, "model": model_key, "source": source, "seed": seed,
        "train_loss": res.training_loss,
        "train_runtime": res.metrics["train_runtime"],
    }

    if source == "imdb":
        _, en_test = load_imdb(1000)
        en_test = en_test.shuffle(seed=42).select(range(2000))
        en_test = en_test.map(tok_fn(tokenizer), batched=True)
        yt, yp, _ = predict(model, tokenizer, en_test)
        out["imdb_test"] = evaluate(yt, yp, f"{tag} / IMDB test (EN)")

    yt, yp, pr = predict(model, tokenizer, marc_val)
    out["marc_ja"] = evaluate(yt, yp, f"{tag} / MARC-ja")
    n_mis = save_misclassified(marc_val, yt, yp, pr, f"results/mis_{tag}_marc.json")
    out["marc_ja"]["n_misclassified"] = n_mis

    if source == "imdb":
        yt, yp, pr = predict(model, tokenizer, wrime)
        out["wrime"] = evaluate(yt, yp, f"{tag} / WRIME")
        save_misclassified(wrime, yt, yp, pr, f"results/mis_{tag}_wrime.json")

    out["total_time"] = time.time() - t0
    with open(f"results/{tag}.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  -> results/{tag}.json ({out['total_time']:.0f}s)")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_monolingual(seed, marc_val):
    tag = f"tohoku_marc_seed{seed}"
    t0 = time.time()

    train_ds = load_marc_ja_train(N_TRAIN, balanced=True)
    model, tokenizer, res = train_model(TOHOKU, train_ds, seed, tag)

    yt, yp, pr = predict(model, tokenizer, marc_val)
    out = {
        "tag": tag, "model": "tohoku-BERT", "source": "marc-ja", "seed": seed,
        "train_loss": res.training_loss,
        "train_runtime": res.metrics["train_runtime"],
        "marc_ja": evaluate(yt, yp, f"{tag} / MARC-ja"),
    }
    out["total_time"] = time.time() - t0

    with open(f"results/{tag}.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  -> results/{tag}.json ({out['total_time']:.0f}s)")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print(f"CUDA: {torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    marc_val = load_marc_ja_val()
    wrime = load_wrime_binary()
    print(f"MARC-ja val: {len(marc_val)} / WRIME: {len(wrime)}")

    all_results = []
    for seed in SEEDS:
        for model_key in MODELS:
            all_results.append(run_transfer(model_key, "imdb", seed, marc_val, wrime))
        for model_key in MODELS:
            all_results.append(run_transfer(model_key, "amazon", seed, marc_val, wrime))
        all_results.append(run_monolingual(seed, marc_val))

    with open("results/ALL.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\n完了：results/ALL.json")
