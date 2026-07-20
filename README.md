# thesis-jlft-csa

修士論文テーマ：**多言語事前学習モデルによる日英感情分析の言語間転移における有効性と限界の分析**

- 著者：王天宇（オウ テンウ）
- 所属：国士舘大学大学院 工学研究科 応用情報工学専攻 M2
- 指導教員：高橋先生
- 期間：2026年4月 〜 2027年1月

---

## 研究概要

mBERT および XLM-RoBERTa を用いて、英語感情分析タスクで学習したモデルを日本語へゼロショット転移した際の有効性と限界を分析する。日本語特有の4つの言語現象（文末否定、オノマトペ、皮肉、混合感情）について誤り分析を行い、Few-shot 適応による改善可能性を検証する。

## 実験設計（3段階）

| Step | 内容 | 状態 |
|---|---|---|
| Step 1 | 準備期（データセット確定、環境構築） | 完了（2026/05） |
| Step 2 | 第1段階実験（ゼロショット転移のベースライン確立） | 完了（2026/07） |
| Step 3 | 第2段階実験（4カテゴリ誤り分析） | 準備中（2026/08〜） |

## 主な結果（Step 2）

3モデル × 3シード（42, 123, 456）、計15実験の結果：

| モデル | ソース | MARC-ja Macro-F1 | WRIME Macro-F1 |
|---|---|---|---|
| mBERT | IMDB | 0.6864 ± 0.0247 | 0.7013 ± 0.0103 |
| XLM-R | IMDB | **0.8812 ± 0.0050** | 0.7917 ± 0.0149 |
| mBERT | Amazon | 0.6220 ± 0.0314 | ─ |
| XLM-R | Amazon | **0.8921 ± 0.0089** | ─ |
| 東北大学 BERT（日本語直接学習） | MARC-ja | **0.9087 ± 0.0027** | ─ |

主要な知見：

- XLM-R のゼロショット性能は、日本語単言語モデルの **97%** に到達（差 2.75ポイント）
- 同じ多言語モデルでも mBERT は 22.2ポイント の大きなギャップ（トークナイズ粒度の差が要因と推定）

## 使用データセット

| 用途 | データセット | 言語 | ドメイン | 規模 |
|---|---|---|---|---|
| ソース（主実験） | IMDB | 英語 | 映画レビュー | 25,000 |
| ソース（補助実験） | Amazon Polarity | 英語 | 商品レビュー | 25,000 |
| ターゲット（主） | MARC-ja | 日本語 | 商品レビュー | 5,654（validation） |
| ターゲット（補助） | WRIME ver2 | 日本語 | SNS投稿 | 1,781（test、2値化後） |

ラベル定義は全データセットで **negative=0, positive=1** に統一。

## ディレクトリ構成

```
thesis-jlft-csa/
├── run_all.py           # Step 2 メイン実験スクリプト（全15実験）
├── src/
│   ├── data.py          # データ読み込み・ラベル統一
│   └── evaluate.py      # 評価関数（Macro-F1, P/R, Confusion Matrix）
├── results/             # 実験結果（JSON）+ 誤分類サンプル
├── logs/                # 実行ログ
└── env/                 # conda 環境定義
```

## 実行方法

```bash
# 環境構築
pip install "datasets<3.0" "transformers<5" accelerate scikit-learn
pip install fugashi unidic-lite   # 東北大BERT の日本語トークナイザ用

# 全実験の実行
python run_all.py 2>&1 | tee logs/run_all.log
```

結果は `results/` に保存される：

- `{model}_{source}_seed{n}.json` — 各実験の評価指標
- `mis_{model}_{source}_seed{n}_{dataset}.json` — 誤分類サンプル（確信度降順）
- `ALL.json` — 全実験の集約

## 実験環境

- ローカル：MacBook Pro (M4 Pro, 24GB)、MPS
- GPU実験：vast.ai / NVIDIA RTX PRO 6000 Blackwell Server Edition

## Author

WANG TIANYU / オウ テンウ
M2, 国士舘大学
**GitHub**: [@outenu](https://github.com/outenu)
