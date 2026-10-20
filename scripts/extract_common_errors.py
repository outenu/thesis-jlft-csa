"""
3シード共通の誤分類を抽出し、アノテーション用シートを作成する。

使い方（~/Projects/thesis で実行）:
    python scripts/extract_common_errors.py

入力:  results/mis_{model}_imdb_seed{42,123,456}_marc.json
出力:  annotation/xlmr_common.csv   XLM-R の3シード共通誤り（本アノテーション対象）
       annotation/pilot.csv         xlmr_common から無作為に40件（予備試行用）
       annotation/mbert_only.csv    mBERT のみの3シード共通誤りから無作為に100件（比較用）
       annotation/summary.json      件数の集計
"""
import csv
import json
import random
from pathlib import Path

RESULTS = Path("results")
OUT = Path("annotation")
SEEDS = [42, 123, 456]
SOURCE = "imdb"
DATASET = "marc"
N_PILOT = 40
N_MBERT = 100
LABEL = {0: "negative", 1: "positive"}

COLUMNS = [
    "id", "sentence", "true_label", "pred_label", "mean_confidence", "also_mbert",
    "main_category", "sub_category", "evidence", "hesitation", "seconds", "memo",
]


def load_errors(model):
    """各シードの誤分類を {文: (正解, 予測, 確信度)} で読み込む"""
    per_seed = []
    for seed in SEEDS:
        path = RESULTS / f"mis_{model}_{SOURCE}_seed{seed}_{DATASET}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        per_seed.append({
            r["sentence"]: (r["true_label"], r["pred_label"], r["confidence"])
            for r in rows
        })
        print(f"  {path.name}: {len(rows)} 件")
    return per_seed


def common(per_seed):
    """全シードで誤分類された文だけを残す"""
    keys = set(per_seed[0])
    for d in per_seed[1:]:
        keys &= set(d)
    out = {}
    for s in keys:
        t, p, _ = per_seed[0][s]
        conf = sum(d[s][2] for d in per_seed) / len(per_seed)
        out[s] = (t, p, conf)
    return out


def write_sheet(path, items, mbert_keys, prefix):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i, (s, (t, p, c)) in enumerate(items, 1):
            w.writerow({
                "id": f"{prefix}{i:03d}",
                "sentence": s,
                "true_label": LABEL[t],
                "pred_label": LABEL[p],
                "mean_confidence": f"{c:.4f}",
                "also_mbert": "Y" if s in mbert_keys else "N",
            })


def main():
    OUT.mkdir(exist_ok=True)

    print("XLM-R")
    xlmr = common(load_errors("XLM-R"))
    print("mBERT")
    mbert = common(load_errors("mBERT"))

    both = set(xlmr) & set(mbert)
    mbert_only = {s: v for s, v in mbert.items() if s not in xlmr}

    xlmr_sorted = sorted(xlmr.items(), key=lambda kv: -kv[1][2])
    write_sheet(OUT / "xlmr_common.csv", xlmr_sorted, set(mbert), "X")

    rng = random.Random(42)
    pilot = rng.sample(xlmr_sorted, min(N_PILOT, len(xlmr_sorted)))
    write_sheet(OUT / "pilot.csv", pilot, set(mbert), "P")

    mbert_sample = rng.sample(sorted(mbert_only.items()), min(N_MBERT, len(mbert_only)))
    write_sheet(OUT / "mbert_only.csv", mbert_sample, set(mbert), "M")

    def direction(d):
        fp = sum(1 for t, p, _ in d.values() if t == 0 and p == 1)
        fn = sum(1 for t, p, _ in d.values() if t == 1 and p == 0)
        return {"negative_to_positive": fp, "positive_to_negative": fn}

    summary = {
        "xlmr_common": len(xlmr),
        "mbert_common": len(mbert),
        "both_models": len(both),
        "mbert_only": len(mbert_only),
        "xlmr_only": len(xlmr) - len(both),
        "xlmr_direction": direction(xlmr),
        "mbert_direction": direction(mbert),
        "pilot": len(pilot),
        "mbert_sample": len(mbert_sample),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"XLM-R 3シード共通誤り : {summary['xlmr_common']} 件")
    print(f"mBERT 3シード共通誤り : {summary['mbert_common']} 件")
    print(f"両モデル共通          : {summary['both_models']} 件")
    print(f"XLM-R のみ            : {summary['xlmr_only']} 件")
    print(f"mBERT のみ            : {summary['mbert_only']} 件")
    print(f"XLM-R 誤りの方向      : {summary['xlmr_direction']}")
    print(f"mBERT 誤りの方向      : {summary['mbert_direction']}")
    print(f"出力先: {OUT}/")


if __name__ == "__main__":
    main()
