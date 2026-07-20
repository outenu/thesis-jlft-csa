from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
)

LABEL_NAMES = ["negative", "positive"]  # 0, 1


def evaluate(y_true, y_pred, title=""):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n--- {title} ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:>9s}  P={p[i]:.4f}  R={r[i]:.4f}  F1={f1[i]:.4f}  n={sup[i]}")
    print("  Confusion Matrix (rows=true, cols=pred):")
    print(f"    {cm.tolist()}")

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": {
            name: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support": int(sup[i]),
            }
            for i, name in enumerate(LABEL_NAMES)
        },
        "confusion_matrix": cm.tolist(),
    }
