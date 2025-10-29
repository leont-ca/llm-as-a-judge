# LLM-as-a-Judge 評価スクリプト

LLM-as-a-JudgeとRagasフレームワークを使用した、ReActチャットボットの応答を自動評価するためのPythonスクリプトです。

## 概要

このリポジトリには、**3つの異なる評価ツール**が含まれています：

1.  **`llm_judge_evaluator.py`**: 詳細な5メトリックのルーブリックを持つカスタムLLM-as-a-Judge
2.  **`ragas_llm_judge_evaluator.py`**: Ragasフレームワークベースの評価
3.  **`format_clarity_evaluator.py`**: Claude 3.5とClaude 4.5 Sonnetの応答のフォーマット/スタイルの類似性を比較

すべてのスクリプトは、LLM-as-a-Judgeとして**Azure OpenAI (GPT-5)とStandard OpenAI**をサポートしています。

## 目次

  - [Installation](#installation)
  - [API Requirements by Program](#api-requirements-by-program)
  - [Custom LLM Judge Evaluator](#custom-llm-judge-evaluator)
  - [Ragas-Based Evaluator](#ragas-based-evaluator)
  - [Format Clarity Evaluator](#format-clarity-evaluator)
  - [Comparison: Which to Use?](#comparison-which-to-use)

-----

## インストール

このリポジトリ内のすべてのスクリプトは、同じ依存関係を共有しています。

1.  このリポジトリをクローンまたはダウンロードします
2.  依存関係をインストールします：

<!-- end list -->

```bash
pip install -r requirements.txt
```

3.  API認証情報を安全に設定します（いずれかの方法を選択）：

### `.env` ファイル

プロジェクトのルートディレクトリに `.env` ファイルを作成します：

**Azure OpenAI (GPT-5) の場合：**

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-actual-azure-key
MODEL_NAME=gpt-5
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

**Standard OpenAI の場合：**

```env
OPENAI_API_KEY=your-actual-openai-key
MODEL_NAME=gpt-4.1
```

## プログラム別のAPI要件

各評価スクリプトは設定オプションに重要な違いがあります。

### モデルの互換性と推奨事項

| スクリプト | 推奨モデル |
|--------|------------------|
| `llm_judge_evaluator.py` | **GPT-5** |
| `ragas_llm_judge_evaluator.py` | **GPT-4** |
| `format_clarity_evaluator.py` | **GPT-5** |

### Temperature（温度）設定

3つのスクリプトすべてに、モデルに基づいてTemperatureを異なる方法で処理する条件付きロジックがあります：

#### **GPT-5** (Azure OpenAI)

  -  **Temperatureは1.0に固定**（設定不可）
  - 環境変数で `MODEL_NAME=gpt-5` を設定

#### **GPT-4 モデル** (Azure OpenAI または Standard OpenAI)

  - **Temperatureは設定可能**（デフォルト：0.7）
  - `MODEL_NAME=gpt-4.1`（またはお好みのGPT-4バリアント）を設定

###  推奨理由

#### **GPT-5を使用する場合：**

  -  **`llm_judge_evaluator.py`** - カスタムルーブリック評価は、GPT-5の強化された機能の恩恵を受けます
  -  **`format_clarity_evaluator.py`** - スタイル/フォーマットの比較は、GPT-5の推論能力と相性が良いです

#### **GPT-4を使用する場合：**

  - **`ragas_llm_judge_evaluator.py`** - Ragasフレームワークは、一貫したメトリクスのために**Temperature制御**を必要とします
      - GPT-4はTemperatureを設定できます（デフォルト0.7）
      - GPT-5の固定Temperature=1.0は、コード実行時に問題を引き起こします
      - **重要**：このスクリプトには `gpt-4.1` のようなGPT-4バリアントを使用してください

## Custom LLM Judge Evaluator

`llm_judge_evaluator.py` スクリプトは、カスタムの5メトリックルーブリックを使用して、包括的でReAct固有の評価を提供します。

### 主な機能

  - **ReAct固有のメトリクス**：思考プロセス、検索品質、引用の正確性を評価
  - **詳細な根拠**：各スコアに対する qualitative な説明を提供
  - **カスタムルーブリック**：ReActチャットボット評価用に調整された採点システム
  - **構造化された出力**：スコアと根拠をCSV形式で整理

### 入力CSVの形式

入力CSVファイルにはヘッダー行が必要で、この順序で正確に3つの列が含まれている必要があります：

1.  **Question**: 元のユーザーの質問
2.  **Model A Response**: モデルAの完全な応答
3.  **Model B Response**: モデルBの完全な応答

### 使用方法

####  コストに関する警告

**このスクリプトはCSVの1行ごとに1回のAPIコールを行います。** 

**常に少量のサンプルで最初にテストしてください：**

```bash
# 最初に5行だけでテストします（推奨！）
python llm_judge_evaluator.py my_test_data.csv -n 5
```

**基本的な使用方法：**

```bash
python llm_judge_evaluator.py my_test_data.csv
```

**カスタム出力ファイルを指定：**

```bash
python llm_judge_evaluator.py my_test_data.csv -o my_results.csv
```

**最初のN行のみを処理（コスト管理）：**

```bash
# 最初の10行のみを処理
python llm_judge_evaluator.py my_test_data.csv -n 10

# 最初の50行をカスタム出力で処理
python llm_judge_evaluator.py my_test_data.csv -n 50 -o test_results.csv
```

### 出力CSVの形式

出力ファイル（デフォルトは `evaluation_output.csv`）には以下が含まれます：

  - すべての元の列（Question, Model\_A\_Response, Model\_B\_Response）
  - 各モデル（AとB）について：
      - Citation Score & Justification
      - Relevance Score & Justification
      - ReAct Performance Thought Score & Justification
      - RAG Retrieval Observation Score & Justification
      - Information Integration Score & Justification
  - Evaluation\_Error 列（評価が失敗した場合のエラーメッセージを含む）

### 採点ルーブリック

各応答は、5つの側面について1～5のスケールで評価されます：

#### 1\. RAG Generation - Citation (1-5)

引用の品質、正確性、必要性を評価します。

#### 2\. Relevance (1-5)

回答がユーザーのクエリ全体にどれだけうまく対応しているかを評価します。

#### 3\. ReAct Performance - Thought (1-5)

推論プロセスの論理的な品質と効率を評価します。

#### 4\. RAG Retrieval - Observation (1-5)

検索されたソース資料の品質と関連性を評価します。

#### 5\. RAG Generation - Information Integration (1-5)

モデルが検索された情報をどれだけ正確に統合しているかを評価します。

-----

## Ragas-Based Evaluator

`ragas_llm_judge_evaluator.py` スクリプトは、Ragasフレームワークと自動ReActログ解析を使用して、最新の標準化された評価アプローチを提供します。

**推奨モデル：GPT-4**（例：`gpt-4.1`） - **一貫したRagasメトリクスのためにTemperature制御が必要です！**（詳細は[API Requirements](#api-requirements-by-program)を参照）

### 主な機能

  - **自動ログ解析**：生のReActログから最終回答（Final Answer）とコンテキスト（Contexts）を自動的に抽出
  - **標準化されたメトリクス**：Ragasのメトリクスの1つである faithfulness を使用
  - **手動でのデータ準備不要**：生のチャットボット出力ログを直接処理
  - **並列比較**：カスタムジャッジ評価との比較が可能

### Ragas用の入力CSV形式

入力CSVには**ヘッダー行**が必要で、正確に3つの列が含まれている必要があります：

1.  **Question**: 元のユーザーの質問
2.  **Model\_A\_Response**: モデルAの完全な**フォーマット済み**ReActログ
3.  **Model\_B\_Response**: モデルBの完全な**フォーマット済み**ReActログ

#### 期待されるReActログの構造（フォーマット済み - `log-output-simplifier.py` を使用）

スクリプトは、これらのセクションを持つログを期待します：

```
## 📝 Task タスク
---
情報検索

## 💬 Reaction 反応
---
None

## 📂 Classification 分類
---
社内

## 📊 Status 状態
---
独立

## 🤖 LLM Thought Process 思考
---
[Reasoning about the task...]

## ⚡ Action 行動
---
社内検索

## ⌨️ Action Input 行動入力
---
[Search query]

## 📚 Raw Search Results (Cleaned) 観察
---
### Result 1
[First search result content...]

## 🔗 URLs URL
---
https://example.com/doc1
関連度：0.95
################################################
### Result 2
[Second search result content...]

## 🔗 URLs URL
---
https://example.com/doc2
関連度：0.90
################################################

## ✅ Final Answer 回答
---
[The final answer to the user's question...]
```

### 使用方法

**基本的な使用方法：**

```bash
python ragas_llm_judge_evaluator.py test_5_rows.csv
```

**カスタム出力ファイルを指定：**

```bash
python ragas_llm_judge_evaluator.py my_data.csv -o ragas_results.csv
```

**最初のN行でテスト：**

```bash
# 最初に3行だけでテスト
python ragas_llm_judge_evaluator.py my_data.csv -n 3
```

### 出力CSVの形式

出力ファイル（デフォルトは `ragas_evaluation_output.csv`）には以下が含まれます：

  - **元の列**：Question, Model\_A\_Response, Model\_B\_Response
  - **解析された列**：
      - `model_A_answer`: モデルAから抽出された最終回答
      - `model_A_contexts`: モデルAのログから抽出されたコンテキストのリスト
      - `model_B_answer`: モデルBから抽出された最終回答
      - `model_B_contexts`: モデルBのログから抽出されたコンテキストのリスト
  - **モデルAのRagasスコア**：
      - `Model_A_faithfulness_score`: コンテキストとの事実上の一貫性 (0-1)
  - **モデルBのRagasスコア**：Model\_Bプレフィックス付きの同じメトリクス
      - `Model_A_faithfulness_score`: コンテキストとの事実上の一貫性 (0-1)

### Ragasメトリクスの説明

#### [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) (0-1)

回答が、与えられたコンテキストと事実上整合しているかを測定します。スコアが高いほど、回答がハルシネーション（幻覚）を起こさず、検索された情報に忠実であることを示します。

-----

## Format Clarity Evaluator

`format_clarity_evaluator.py` スクリプトは、Claude 4.5 Sonnetの応答フォーマットが（ゴールデンスタンダードとして使用される）Claude 3.5 Sonnetのフォーマットにどれだけ近いかを評価するための専用ツールです。

### 主な機能

  - **フォーマット比較**：マークダウン、リスト、構造の類似性を評価
  - **単一スコア**：包括的なフォーマット/明瞭性スコア（1-5）を1つ提供
  - **自動解析**：生のReActログから "Final Answer" セクションを抽出
  - **詳細な根拠**：フォーマットが一致する、または異なる理由を説明

### 評価対象

エバリュエーターは以下に焦点を当てます：

  - マークダウンの使用（`##` のような見出し、`**text**` のような太字）
  - リスト構造（`-` の箇条書き vs `1.` の番号付き）
  - アイデアの論理的な分離（段落、セクション）
  - 全体的な構造の類似性

### 入力CSVの形式

入力CSVファイルにはヘッダー行が必要で、この順序で正確に3つの列が含まれている必要があります：

1.  **Question**: 元のユーザーの質問
2.  **Model A Response**: モデルAの完全な応答
3.  **Model B Response**: モデルBの完全な応答

### 使用方法

**基本的な使用方法：**

```bash
python format_clarity_evaluator.py input.csv
```

**行数を制限してテスト：**

```bash
# 最初の5行のみを処理
python format_clarity_evaluator.py input.csv -n 5

# 最初の10行をカスタム出力で処理
python format_clarity_evaluator.py input.csv -n 10 -o test_results.csv
```

**カスタム出力ファイルを指定：**

```bash
python format_clarity_evaluator.py input.csv -o my_format_results.csv
```

### 出力CSVの形式

出力ファイル（デフォルトは `format_clarity_output.csv`）には以下が含まれます：

| 列 | 説明 |
|--------|-------------|
| `Question` | 元の質問 |
| `Claude_3.5_Final_Answer` | Claude 3.5のログから解析された最終回答 |
| `Claude_4.5_Final_Answer` | Claude 4.5のログから解析された最終回答 |
| `Format_Clarity_Score` | 1～5のスコア |
| `Format_Clarity_Justification` | スコアの詳細な説明 |
| `Evaluation_Error` | 評価が失敗した場合のエラーメッセージ |

### 採点ルーブリック

LLMジャッジは詳細な5段階のスケールを使用します：

  - **5 (優秀)**：ほぼ同一のフォーマット。マークダウン、リスト、構造を完璧に反映している
  - **4 (良い)**：ほとんど同様だが、わずかな逸脱がある（例：箇条書き vs 番号付き）
  - **3 (許容)**：いくつかの類似点はあるが、重大な違いもある（例：リスト vs 段落）
  - **2 (悪い)**：ほとんどが異なり、3.5モデルの構造に似ていない
  - **1 (非常に悪い)**：完全に異なるフォーマット（例：構造化 vs 単一のテキストブロック）

### 出力例

処理後、スクリプトは要約統計を表示します：

```
✓ 評価完了！
✓ 結果を format_clarity_output.csv に書き込みました
✓ 50行を処理しました

📊 平均フォーマット明瞭性スコア： 3.84/5.0
📊 スコア分布：
5    12
4    21
3    15
2     2
1     0
```

-----

## エラーハンドリング

すべてのスクリプトには、堅牢なエラーハンドリングが含まれています：

  - **APIエラー**：指数関数的バックオフによる自動リトライ（最大3回）
  - **JSON解析エラー**：応答の形式が不正な場合に検証し、リトライします
  - **APIキーの欠落**：セットアップ手順を含む明確なエラーメッセージ
  - **ファイルが見つからない**：入力ファイルが見つからない場合のグレースフルなエラーハンドリング

すべてのエラーは、出力CSVのそれぞれのエラー列に記録されます。

-----

## パフォーマンスに関する考慮事項

  - APIコールはリトライロジックと共に順次実行されます
  - 進捗はtqdmプログレスバーを介して表示されます
  - Temperatureは一貫した評価のために最適化されています
  - Max tokensは各エバリュエーターに適切に設定されています