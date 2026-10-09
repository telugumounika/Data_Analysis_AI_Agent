# Data Analysis AI Agent

Upload a CSV or Excel file and ask questions about it in plain English. The agent picks the right analysis tool, runs it with Pandas, and returns an answer, tables, charts and an explanation of how it got there.

> Full-stack GenAI project: FastAPI + Pandas + tool-calling LLM (local Qwen via LM Studio, or any OpenAI-compatible API) + vanilla HTML/CSS/JS dashboard.

## Features
- CSV / Excel upload with type, size and empty-file validation
- Dashboard: rows, columns, missing values, duplicates, column types, statistics, preview
- Natural-language questions: totals, group-by, top/bottom N, filters, trends, correlations, outliers
- Charts: bar, line, pie, histogram, scatter, correlation heatmap
- **Fill missing values, with a question first**: say "fill missing values" and the agent asks which column and which method (mean, median, mode, forward fill, backward fill, interpolation, your own value, or delete rows). Nothing changes until you press Apply.
- **Edit data tab**: edit cells like a spreadsheet, delete rows, undo (5 levels), change history, download the edited CSV/Excel, or replace the file with a newer version
- **Auto re-analysis**: after any change the dashboard, AI insights and your last 3 analyses are re-run automatically; older answers are greyed out as "before the update"
- Live LLM status badge (connected / model not found / offline)
- AI insights built only from pre-computed facts (no invented numbers)
- Chat with follow-up memory, plus a "How I got this" panel showing the tools used
- **Safe by design**: the LLM can only call 13 whitelisted functions; it never executes code
- Automatic fallback to a rule-based analyst if the LLM is off or fails

## Architecture
```
Browser (HTML/CSS/JS)
   │  fetch() JSON / FormData
   ▼
FastAPI  (main.py)
   │
   ├─ file_handler.py  validate + load DataFrame
   ▼
Agent (agent.py) ──► LLM (llm_client.py → LM Studio / Qwen or cloud API)
   │   ▲  picks a tool, reads the result, writes the answer
   ▼   │
Tools (tools.py)  whitelist of Pandas functions ──► charts.py (Matplotlib → base64 PNG)
   │
   └─ planner.py  rule-based fallback    insights.py  facts → insights
```

## Run it (Windows, VS Code terminal)
```powershell
cd data-analysis-ai-agent\backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```
Open http://localhost:8000 and upload `sample_data/sales_data.csv`.

If you see *"LLM unavailable… Cannot connect"*, the LM Studio server isn't running. Either start it (Developer tab → Start Server) or set `USE_LLM=false` in `.env`. The badge at the top right of the page shows the live status. The app works either way.

### Enable the local Qwen model (LM Studio)
1. In LM Studio, download a Qwen instruct model that supports tool use (e.g. Qwen2.5-7B-Instruct).
2. Open the **Developer** tab → load the model → **Start Server** (port 1234).
3. Copy the model's identifier into `LLM_MODEL` in `backend/.env`.
4. Set `USE_LLM=true` and restart uvicorn.

### Use a cloud API instead
Set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` in `.env` (see `.env.example`). Never commit `.env`.

## Tests
```powershell
cd backend
pytest -q
```
62 tests cover uploads, tools, charts, agent logic, missing-value filling, editing, undo, download, LLM failure, and prompt-injection safety.

## Project structure
```
data-analysis-ai-agent/
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI routes
│   │   ├── config.py        settings from .env
│   │   ├── file_handler.py  upload validation + loading
│   │   ├── tools.py         analysis tools + whitelist + schemas
│   │   ├── cleaning.py      fill-missing methods, cell edits, row deletes
│   │   ├── charts.py        Matplotlib/Seaborn charts
│   │   ├── llm_client.py    HTTP client for the LLM
│   │   ├── agent.py         tool-calling loop
│   │   ├── planner.py       rule-based fallback
│   │   └── insights.py      fact-based insights
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── frontend/  index.html · style.css · app.js
├── sample_data/  sales_data.csv · generate_sample.py
└── docs/  LEARNING_GUIDE.md · INTERVIEW_PREP.md
```

## Future improvements
- Persist datasets/sessions in SQLite; user accounts
- Streaming responses; export report to PDF
- Multi-step planning for complex questions; Plotly interactive charts
- Docker image and cloud deployment

## Screenshots
Add screenshots of the dashboard and chat here (`docs/screenshots/`).
