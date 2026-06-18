# FinSight

LLM-augmented analysis tool for SEC financial filings (10-K / 10-Q). Ask questions about filings in plain English, extract structured financial signals, compare filings across years, and correlate signals with forward stock returns.

## Features

- **RAG Q&A** — ask questions across all loaded filings; every answer includes numbered citations and a hallucination audit
- **Signal extraction** — revenue guidance, capex guidance, risk-factor count, and FinBERT sentiment per filing
- **Fundamentals** — computed ratios (margins, leverage, FCF), YoY growth, and health flags from SEC XBRL or yfinance
- **Filing diff** — detects substantive disclosure changes between two filings of the same company
- **Signal → returns** — OLS regression of sentiment signals against 5- and 20-day forward stock returns, with SPY market control

---

## Setup

### Prerequisites

- Python 3.10 or higher
- Git

---

### Linux / macOS

```bash
# 1. Clone the repository
git clone https://github.com/superxz3r0/LLM-Augmented-Financial-Report-Analysis-Tool.git
cd LLM-Augmented-Financial-Report-Analysis-Tool

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install PyTorch CPU build to avoid a large GPU download
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Run the app
streamlit run app.py
```

---

### Windows

```powershell
# 1. Clone the repository
git clone https://github.com/superxz3r0/LLM-Augmented-Financial-Report-Analysis-Tool.git
cd LLM-Augmented-Financial-Report-Analysis-Tool

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install PyTorch CPU build
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## LLM API Keys (optional)

Without a key the app runs in **extractive mode** — retrieved passages are shown verbatim with citations. To enable generated answers, paste a key in the sidebar after launching the app:

| Provider | Where to get it | Model used |
|---|---|---|
| Google Gemini | [aistudio.google.com](https://aistudio.google.com) (free tier) | gemini-2.5-flash |
| OpenAI | [platform.openai.com](https://platform.openai.com) | gpt-4o-mini |

Keys are stored locally in `.streamlit/secrets.toml` and never sent anywhere except the respective API.

---

## Fetching real SEC filings

The bundled `data/sample/` folder contains two synthetic companies for offline demo. To load real EDGAR data:

```bash
# Linux / macOS
python scripts/fetch_filings.py --tickers AAPL MSFT NVDA --forms 10-K 10-Q --years 3

# Windows
python scripts\fetch_filings.py --tickers AAPL MSFT NVDA --forms 10-K 10-Q --years 3
```

Files are saved to `data/filings/` as `<TICKER>_<FORM>_<YYYY-MM-DD>.txt`. The app picks them up automatically on the next restart.

To fetch the full default universe of 24 companies (≈300 filings):

```bash
python scripts/fetch_filings.py
```

> Requires `edgartools` (already in `requirements.txt`). The SEC requires a declared identity; set it via the `EDGAR_IDENTITY` environment variable or leave the default.

---

## Running tests

```bash
# Linux / macOS
pytest -q

# Windows
pytest -q
```

All tests run offline using the bundled sample data — no internet or API keys required.

To run the extraction accuracy evaluation:

```bash
# Linux / macOS
python eval/run_extraction_eval.py

# Windows
python eval\run_extraction_eval.py
```

---

## Project structure

```
finsight/
├── app.py                     # Streamlit UI
├── requirements.txt
├── Dockerfile
├── src/finsight/
│   ├── config.py              # Central settings and paths
│   ├── ingest.py              # Filing parser (HTML + plain text)
│   ├── store.py               # SQLite signal cache
│   ├── chunker.py             # Sentence-aware text chunker
│   ├── index.py               # Hybrid BM25 + vector retrieval
│   ├── rag.py                 # RAG answering with citations
│   ├── audit.py               # Hallucination audit
│   ├── extract.py             # Signal extraction (4 signal types)
│   ├── sentiment.py           # FinBERT sentiment scoring
│   ├── diff.py                # Substantive filing diff engine
│   ├── metrics.py             # Financial ratios and health flags
│   ├── xbrl.py                # SEC XBRL fundamentals fetcher
│   └── returns.py             # Signal → forward return regression
├── scripts/
│   └── fetch_filings.py       # EDGAR filing downloader
├── eval/
│   ├── run_extraction_eval.py # Extraction accuracy evaluation
│   └── labels.json            # Hand-labelled held-out set
├── tests/
│   └── test_pipeline.py       # Unit tests (offline)
└── data/
    ├── sample/                # Bundled synthetic filings
    ├── filings/               # Real EDGAR downloads (gitignored)
    └── index/                 # Persisted vector index (gitignored)
```

---

## Docker

```bash
docker build -t finsight .
docker run -p 8501:8501 finsight
```

Access the app at `http://localhost:8501`.
