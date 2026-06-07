import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from src.data import load_imdb, load_marc_ja, load_wrime_binary
from src.evaluate import evaluate

def run_experiment(model_name, label, n_train=None, seed=42):
    print(f"  {label}: {model_name}")
    print(f"  学習データ: IMDB {'(subset=' + str(n_train) + ')' if n_train else '(full)'}")
    print(f"  seed: {seed}")
    
    print("\n[1/5] データ読み込み...")
    train_ds, test_ds_en = load_imdb(n_samples=n_train)
    marc_ja = load_marc_ja()
    wrime = load_wrime_binary()
    print(f"  IMDB train: {len(train_ds)}")
    print(f"  IMDB test:  {len(test_ds_en)}")
    print(f"  MARC-ja:    {len(marc_ja)}")
    print(f"  WRIME:      {len(wrime)}")

    print("\n[2/5] トークナイズ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_imdb(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    def tokenize_ja(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=256)

    train_ds = train_ds.map(tokenize_imdb, batched=True)
    test_ds_en = test_ds_en.map(tokenize_imdb, batched=True)
    marc_ja = marc_ja.map(tokenize_ja, batched=True)
    wrime = wrime.map(tokenize_ja, batched=True)

    print("\n[3/5] ファインチューニング...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    args = TrainingArguments(
        output_dir=f"./tmp_{label}_{seed}",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    result = trainer.train()
    print(f"  訓練損失: {result.training_loss:.4f}")
    print(f"  訓練時間: {result.metrics['train_runtime']:.1f}秒")

    print("\n[4/5] 推論...")
    device = model.device
    model.eval()

    def predict(dataset):
        all_preds = []
        all_labels = []
        batch_size = 32
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            inputs = tokenizer(
                batch["text"] if "text" in batch else batch["sentence"],
                return_tensors="pt", padding=True, truncation=True, max_length=256
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"])
        return all_labels, all_preds

    print("\n[5/5] 評価...")

    print("\n英語 IMDB テスト")
    y_true_en, y_pred_en = predict(test_ds_en)
    res_en = evaluate(y_true_en, y_pred_en)

    print("\n日本語 MARC-ja")
    y_true_marc, y_pred_marc = predict(marc_ja)
    res_marc = evaluate(y_true_marc, y_pred_marc)

    print("\n日本語 WRIME")
    y_true_wrime, y_pred_wrime = predict(wrime)
    res_wrime = evaluate(y_true_wrime, y_pred_wrime)

    results = {
        "model": label,
        "seed": seed,
        "n_train": n_train or "full",
        "imdb_test": res_en,
        "marc_ja": res_marc,
        "wrime": res_wrime,
    }
    out_path = f"results/{label}_seed{seed}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果を {out_path} に保存しました")

    return results

if __name__ == "__main__":
    run_experiment("bert-base-multilingual-cased", "mBERT", n_train=5000, seed=42)