from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

def evaluate(y_true, y_pred, label_names=["negative", "positive"]):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    for i, name in enumerate(label_names):
        print(f"  {name:>10s}  P={precision[i]:.4f}  R={recall[i]:.4f}  F1={f1[i]:.4f}  n={support[i]}")
    print(f"\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {
            name: {"precision": precision[i], "recall": recall[i], "f1": f1[i]}
            for i, name in enumerate(label_names)
        },
        "confusion_matrix": cm.tolist()
    }