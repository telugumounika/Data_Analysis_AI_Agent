# Learning Guide — read the project in this order

Each step: read the file, run it, then do the exercise. Don't move on until you can explain the "Be able to say" line.

## 1. Big picture
- **AI agent** = an LLM that can *choose and call tools* and use the results, instead of only chatting.
- **Chatbot** answers from memory (can hallucinate numbers). **Agent** calls Pandas for the real number, then explains it.
- **LLM** understands the question and writes the explanation. **Pandas** does the maths. **FastAPI** connects the browser to Python.

Flow of "Which category has the highest sales?":
1. Browser POSTs the question to `/api/ask`.
2. `agent.py` sends question + column list + tool list to the LLM.
3. LLM replies "call `group_data(group_by=Category, value_column=Sales, agg=sum)`".
4. `tools.py` runs it with Pandas and returns a table.
5. Result goes back to the LLM, which writes the final sentence.
6. Browser shows answer, table, and "How I got this".

## 2. Files in learning order
| Order | File | Concept | Exercise |
|---|---|---|---|
| 1 | `sample_data/sales_data.csv` | Rows, columns, dtypes | Open in Excel, list the columns and types |
| 2 | `backend/app/file_handler.py` | Validation, `pd.read_csv/read_excel`, DataFrame | Add a rule that rejects files with more than 100 columns |
| 3 | `backend/app/tools.py` (first 5 tools) | `groupby`, `sum`, `describe`, `isna` | Add a tool `count_unique(column)` |
| 3b | `backend/app/cleaning.py` | `fillna`, `ffill`, `bfill`, `interpolate`, returning a *new* DataFrame so undo is easy | Add a method "fill with previous month's average" |
| 4 | `backend/app/charts.py` | Matplotlib, base64 images | Change the chart colour |
| 5 | `backend/app/main.py` | GET/POST, JSON, FormData, CORS, status codes | Add `GET /api/datasets` that lists uploaded names |
| 6 | `frontend/app.js` | `fetch`, DOM, async/await | Show a spinner while uploading |
| 7 | `backend/app/planner.py` | Question → tool mapping (no AI) | Teach it "lowest" and "median" questions |
| 8 | `backend/app/llm_client.py` | Prompt, temperature, HTTP to an LLM | Print the raw reply from LM Studio |
| 9 | `backend/app/agent.py` | System prompt, tool calling loop, fallback | Lower `MAX_AGENT_STEPS` to 1 and see what changes |
| 10 | `backend/app/insights.py` | Preventing hallucination | Add a fact: "percent of rows with Profit < 0" |
| 11 | `backend/tests/` | pytest, fixtures, monkeypatching | Write a test for your new tool |

## 3. Key concepts in plain words
- **Virtual environment**: a private folder of packages for this project only, so versions don't clash.
- **Endpoint**: a URL the backend listens on. GET reads, POST sends data.
- **CORS**: browser rule that blocks a page from calling a different origin unless the server allows it.
- **Prompt / system prompt**: the instructions you give the LLM. The system prompt sets its rules.
- **Temperature**: randomness. 0.1 = consistent, good for data work.
- **Tool calling**: you describe functions in JSON; the LLM replies with *which function and arguments*; **your code** runs it.
- **Hallucination**: the model inventing facts. Prevented here by (1) tools for every number, (2) a prompt that forbids guessing, (3) insights built from computed facts.
- **Why not `exec()` generated code?** The model could be tricked (prompt injection) into deleting files or leaking data. A whitelist of functions means the worst it can do is call a harmless analysis tool with odd arguments, which we validate.

## 4. Security checklist (already implemented — find each in the code)
- File extension whitelist and size limit (`file_handler.py`)
- Random server-side filename, file deleted after load
- Tool whitelist and argument type filtering (`run_tool`)
- Column names validated against the real DataFrame (`resolve_column`)
- Row limits on results, question length limit
- Generic 500 error (no stack traces to the browser)
- API keys only in `.env`, which is git-ignored

## 5. Ideas to extend (pick one and own it)
1. New tool: `compare_groups(col, a, b)`
2. Persist chat history in SQLite
3. Export the conversation as PDF
4. Add a "date range filter" to `filter_data`

## 6. How editing and auto re-analysis work
1. Every change goes through `_commit()` in `main.py`: old DataFrame is pushed on an undo stack, the new one replaces it, `version` goes up.
2. Each change endpoint returns the fresh dashboard data (`build_overview`).
3. The frontend's `onDataUpdated()` re-renders the dashboard, reloads the edit table, regenerates insights and re-asks your last 3 questions.
4. **Why the agent asks before filling:** modifying data is irreversible-feeling and a judgement call (mean vs median vs ffill). So the LLM never does it; `agent.py` detects the intent and returns `needs_input`, the UI shows a form, and only your click calls `/fill-missing`.
