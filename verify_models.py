from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print("=== mBERT ===")
tok1 = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
print("Tokenize 日本語:", tok1.tokenize("この映画は本当に面白かった"))

print("=== XLM-R ===")
tok2 = AutoTokenizer.from_pretrained("xlm-roberta-base")
print("Tokenize 日本語:", tok2.tokenize("この映画は本当に面白かった"))

print("MPS:", torch.backends.mps.is_available())