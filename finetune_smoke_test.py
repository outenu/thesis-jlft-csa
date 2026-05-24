import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

def run_smoke_test(model_name, label):
    print(f"\n{'='*50}")
    print(f"  {label}: {model_name}")
    print(f"{'='*50}")

    ds = load_dataset("imdb", split="train[:1000]")
    ds = ds.rename_column("text", "sentence")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=f"./tmp_{label}",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds)
    result = trainer.train()
    print(f"\n訓練損失: {result.training_loss:.4f}")
    print(f"訓練時間: {result.metrics['train_runtime']:.1f}秒")

    # 日本語 zero-shot 推論テスト
    device = model.device
    model.eval()
    test_texts = ["この映画は本当に面白かった", "最悪の映画だった"]
    inputs = tokenizer(test_texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)
    for text, pred in zip(test_texts, preds):
        print(f"  {text} → {'positive' if pred == 1 else 'negative'}")

run_smoke_test("bert-base-multilingual-cased", "mBERT")
run_smoke_test("xlm-roberta-base", "XLM-R")