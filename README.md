# 日英クロスリンガル感情分析 (JLFT-CSA)

修士論文 (M2) 研究プロジェクト。日本語言語特徴に基づくエラー分析と難易度ガイド型 few-shot 選択戦略を用いたクロスリンガル感情分析。

## 研究概要

- **モデル**: mBERT / XLM-RoBERTa
- **手法**: JLFT-CSA (4カテゴリの日本語言語特徴エラー分析)
  - 文末否定
  - オノマトペ
  - 皮肉
  - 混合感情
- **実験設定**:
  - 主実験: IMDB (英語) → Yahoo映画レビュー (日本語)
  - 補助実験: Amazon English Reviews → MARC-ja

## ディレクトリ構成

```
thesis/
├── src/             # メインソースコード
├── data/
│   ├── raw/         # 生データ
│   └── processed/   # 前処理済みデータ
├── models/          # 学習済みモデル (gitignore)
├── experiments/     # 実験ログ・結果
├── notebooks/       # 探索的分析用 Jupyter notebook
├── docs/            # ドキュメント、研究計画書
├── env/             # 環境設定ファイル
└── scripts/         # シェルスクリプト
```

## 環境セットアップ

```bash
conda env create -f env/thesis.yml
conda activate thesis
```

## Author

WANG TIANYU / オウ テンウ
M2, 国士舘大学
**GitHub**: [@outenu](https://github.com/outenu)
