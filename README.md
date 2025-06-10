# OpenAI Model Comparison Tool

OpenAIの異なるモデルを比較するためのツールです。非同期処理を使用して2つのモデルを並行実行し、レイテンシー、トークン使用量、コストを比較できます。

## 機能

- 2つのモデルを並行実行して比較
- レイテンシー、トークン使用量、コストの計測
- コマンドラインインターフェース（CLI）
- StreamlitベースのWeb UI
- 結果のJSONLログ保存

## インストール

Poetryを使用してインストールします：

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/openai-model-comparison.git
cd openai-model-comparison

# 依存関係のインストール
poetry install
```

## 使用方法

### コマンドライン（CLI）

```bash
# 環境変数の設定
export OPENAI_API_KEY="your-api-key"

# モデル比較の実行
poetry run python test.py -p "こんにちは、自己紹介してください" -m1 gpt-4o -m2 gpt-4.1-nano
```

### Web UI

```bash
# Streamlitアプリの起動
poetry run streamlit run app.py
```

## 設定

### モデルと価格

`test.py` の `PRICING_PER_MILLION` 辞書でモデルと価格を設定できます：

```python
PRICING_PER_MILLION = {
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}
```

### ログ

結果は `logs/run.jsonl` に保存されます。各実行は以下の情報を含むJSONL形式で記録されます：

- タイムスタンプ
- モデル名
- プロンプト
- 応答
- レイテンシー（ミリ秒）
- トークン使用量
- コスト（USD）

## ライセンス

MIT License

## 注意事項

- OpenAI APIキーは環境変数 `OPENAI_API_KEY` で設定してください
- APIキーは公開しないように注意してください
- 価格は変更される可能性があります。最新の価格は[OpenAIの料金ページ](https://openai.com/api/pricing)で確認してください 