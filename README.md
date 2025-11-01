# tcm-retrieval-eval

## 專案簡介
`tcm-retrieval-eval` 提供一套端到端流程，協助針對中醫證候知識庫建立向量索引、執行檢索評估與視覺化結果。專案以 Chroma 物件資料庫為核心，支援多種嵌入模型（OpenAI、vLLM、自行部署的 Qwen3 家族與 HuggingFace 模型）以及Selective欄位選擇策略，可快速比較不同欄位組合、不同向量模型與 reranker 的表現。

## 功能總覽
- `ingest.py`：將 JSON/JSONL 原始資料轉換為 Chroma 向量庫，支援選擇性欄位與批次處理。
- `evaluate_retrieval.py`：載入測試查詢、計算 MRR / Recall@K、支援 Query2Doc、keywords 模式與外部 reranker。
- `generate_charts.py`：從評估結果產生排名分佈、累積分布與 recall 曲線，也支援批次與實驗比較。
- `load_data*.sh`：一鍵建立多組Selective欄位向量庫，方便對照不同資料欄位切分設定。

## 環境需求
- Python 3.10 以上。
- 依照使用的嵌入服務準備對應 API（例如 OpenAI API、vLLM embedding 服務、客製化 Qwen3 embedding 服務、或本機 HuggingFace 模型）。
- （選用）若啟用 reranker，需先啟動對應的 HTTP 服務，預設為 `http://localhost:8001` 並符合 `/health` 與 `/v1/rerank` 介面。
- 建議在專案根目錄建立 `.env` 檔案，放置必要金鑰：
  ```
  OPENAI_API_KEY=your-openai-key
  ```

### 安裝相依套件
#### 使用 uv（推薦）
```bash
pip install uv
uv sync
source .venv/bin/activate
```


## 資料準備
### 症候知識庫（RAG 來源）
- 預設資料位於 `data/syndrome_knowledge_fixed.json`，結構如下（部分欄位）：
  ```json
  {
    "Name": "风寒袭肺证",
    "Definition": "...",
    "Typical_performance": "...",
    "Common_disease": "...",
    "id": 0
  }
  ```
- `ingest.py` 同時支援 JSON 陣列與 JSONL（加入 `--jsonl` 旗標）格式。若採用Selective欄位，可透過 `--fields` 選擇 `Name`, `Definition`, `Typical_performance`, `Common_disease` 等欄位。

### 測試查詢（檢索評估）
- `evaluate_retrieval.py` 預期的測試資料為 JSON 陣列，每筆需至少包含：
  - `user_id`：用於識別查詢。
  - `prompt`：查詢文字。
  - `expected_doc_id`：期望命中的向量文件 `metadata["id"]`。
  - `expected_answer`：便於檢視輸出的標註答案。
  - （選用）`pseudo_document`：若啟用 `--use-pseudo-doc`，會附加在查詢後方。
  - （選用）`keywords`：若啟用 `--use-keywords`，會直接用該欄位做檢索。
- 將測試檔放在 `data/` 下，或透過 `--json-file` 指向自訂路徑。

## 建立 Chroma 向量資料庫
### 基本指令
```bash
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name syndrome_db
```

### 重要參數
- `--json-file`：輸入資料位置（JSON 或 JSONL）。
- `--db-name`：輸出的資料庫名稱，向量資料會儲存在 `chroma_dbs/<db_name>/`。
- `--jsonl`：資料為 JSON Lines 時加上。
- `--jq-schema`：若需要以 jq 的邏輯動態組合欄位，可提供對應表達式。
- `--fields`：Selective模式，使用逗號分隔欄位名稱（如 `Name,Definition`）。不支援的欄位會直接報錯。
- `--embedding-model`：指定嵌入模型，可使用 `provider:model` 或直接填模型名稱。
- `--list-models`：列出所有支援的模型並結束程式。

| Provider      | 模型範例                            | 預設 base_url / 備註                                   |
| ------------- | ----------------------------------- | ------------------------------------------------------ |
| `openai`      | `text-embedding-3-large`            | 需要 `OPENAI_API_KEY`。                                |
| `vllm`        | `Qwen3-Embedding-8B`                | 預設 `http://localhost:8010/v1`。                      |
| `custom`      | `Qwen3-Embedding-0.6B-finetuned` 等 | `ingest.py` 預設 `http://localhost:8000/v1`。          |
| `huggingface` | `BAAI/bge-large-zh-v1.5`            | 使用本地模型，需安裝 `sentence-transformers`。         |

> `evaluate_retrieval.py` 中 `custom` 提供者的預設 base_url 為 `http://localhost:8002/v1`，如部署位置不同請分別調整腳本或新增環境變數。
> Qwen3-Embedding-0.6B-finetuned等系列模型，url可透過[TCMEmbeddingModel](https://github.com/NYCU-CGI-LLM/TCMEmbeddingModel)部署

### 批次腳本
- `./load_data.sh`：以 OpenAI（預設）嵌入建立 13 個欄位組合的資料庫。
- `./load_data_qwen3.sh`：以 vLLM Qwen3-Embedding-8B 建立同樣的Selective欄位資料庫。
- `./load_data_qwen3_finetuned.sh`：針對自行微調的 Qwen3 模型建立資料庫（需先啟動服務）。

## 檢索評估流程
### 基本指令
```bash
python evaluate_retrieval.py \
  --json-file data/tcm_sd_test_rc_direct.json \
  --db-name syndrome_db \
  --num-queries 100 \
  --embedding-model openai:text-embedding-3-large
```

### 常用參數
- `--num-queries`：隨機抽樣的測試筆數。
- `--k`：嵌入檢索回傳的候選數（預設 1027，建議與知識庫大小相符）。
- `--save-top-results`：寫入結果 JSON 的候選筆數。
- `--use-pseudo-doc`：啟用 Query2Doc（`prompt + pseudo_document`）。
- `--use-keywords`：改用 `keywords` 欄位做檢索。
- `--max-concurrent`：異步查詢的最大併發數。
- `--list-models`：同 `ingest.py`，列出支援的嵌入模型。

### 兩階段檢索（Rerank）
- 加上 `--rerank` 後，流程會先以嵌入檢索取 `--rerank-top-n` 筆，再呼叫 `--reranker-api-url` 提供的 reranker API 重排。
- reranker API 需支援：
  - `GET /health`：回應模型與設備資訊（JSON）。
  - `POST /v1/rerank`：body 包含 `model`, `query`, `documents`（陣列），可選 `top_n`。
- 預設模型名稱為 `Qwen/Qwen3-Reranker-0.6B`；如 API 回傳不同名稱會在日誌中警告。

### 輸出內容
- 結果會儲存在 `outputs/run_<db_name>_<num_queries>[_...]`，檔案結構包含：
  - `evaluation_results.json`：每筆查詢的 RR、Recall@K、排名、時間統計與前幾筆檢索結果。
  - `ranking_histogram.png`：期望文件排名分布 + L2 距離趨勢。
  - `ranking_cumulative.png`：排名累積分布與 Recall@K 標記。
  - `recall_curve_simplified.png`：僅顯示 Recall@5/10/20/50/100 的折線圖。
- CLI 會同時輸出整體指標（平均 MRR / Recall@K / 時間統計）與個別查詢摘要。

## 結果視覺化與比較
`generate_charts.py` 可對既有的 `evaluation_results.json` 重新生成圖表，或比較兩組實驗。

```bash
# 單一實驗重新繪圖（輸出至檔案所在目錄）
python generate_charts.py outputs/run_syndrome_db_5486/evaluation_results.json

# 批次處理 outputs 底下的所有實驗
python generate_charts.py --batch outputs/

# 比較兩組實驗
python generate_charts.py --compare outputs/exp1 outputs/exp2
```

在比較模式下會額外產生 `ranking_cumulative_comparison.png`，並於終端列出各個 Recall@K 的差異。


## 資料來源與授權（Data Sources & License）
- 上游來源：Borororo/ZY-BERT（<https://github.com/Borororo/ZY-BERT>）  
  - 取得版本（commit/tag）：`72a89518392ec28f5dc1c8ed252a6c127ca2145b`  
  - 存取日期： `2025-11-01`  
  - 授權：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  - 本專案資料屬 **上游的衍生資料**，以與上游相容之授權釋出；使用與再散布需遵守上游條款（例如：非商業、需標註、相同方式分享等，如適用）。

### 本專案釋出的檔案與處理
| 檔名 | 來源（來自 ZY-BERT） | 我們的修補／轉換 | 可重現指令 |
|---|---|---|---|
| `data/syndrome_knowledge_fixed.json` | `syndrome_knowledge.json` | 將欄位名稱 `Common_isease` 更正為 `Common_disease`|
| `data/tcm_sd_test_rc_direct.json` | `test.json` + `syndrome_vocab.txt` | 使用本庫轉換腳本產生 **direct** 格式 | `python tcm_sd_prompt_generator.py --format direct --input data/test.json --vocab_path data/syndrome_vocab.txt --output data/tcm_sd_test_rc_direct.json` |

> 註：本庫僅釋出**結構化／轉換後**內容；不包含未經授權的原文段落。若需完整原始資料，請依上游規範自行取得。


### 引用（Citations）
- **TCM-SD 論文**  
```bibtex
@inproceedings{mucheng-etal-2022-tcm,
  title = "{TCM}-{SD}: A Benchmark for Probing Syndrome Differentiation via Natural Language Processing",
  author = "Ren, Mucheng  and
    Huang, Heyan  and
    Zhou, Yuxiang  and
    Cao, Qianwen  and
    Bu, Yuan  and
    Gao, Yang",
  booktitle = "Proceedings of the 21st Chinese National Conference on Computational Linguistics",
  month = oct,
  year = "2022",
  address = "Nanchang, China",
  publisher = "Chinese Information Processing Society of China",
  url = "https://aclanthology.org/2022.ccl-1.80",
  pages = "908--920",
  abstract = "{``}Traditional Chinese Medicine (TCM) is a natural, safe, and effective therapy that has spread and been applied worldwide. The unique TCM diagnosis and treatment system requires a comprehensive analysis of a patient{'}s symptoms hidden in the clinical record written in free text. Prior studies have shown that this system can be informationized and intelligentized with the aid of artificial intelligence (AI) technology, such as natural language processing (NLP). However, existing datasets are not of sufficient quality nor quantity to support the further development of data-driven AI technology in TCM. Therefore, in this paper, we focus on the core task of the TCM diagnosis and treatment system{---}syndrome differentiation (SD){---}and we introduce the first public large-scale benchmark for SD, called TCM-SD. Our benchmark contains 54,152 real-world clinical records covering 148 syndromes. Furthermore, we collect a large-scale unlabelled textual corpus in the field of TCM and propose a domain-specific pre-trained language model, called ZYBERT. We conducted experiments using deep neural networks to establish a strong performance baseline, reveal various challenges in SD, and prove the potential of domain-specific pre-trained language model. Our study and analysis reveal opportunities for incorporating computer science and linguistics knowledge to explore the empirical validity of TCM theories.{''}",
  language = "English",
}
```