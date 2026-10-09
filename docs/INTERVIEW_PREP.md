# Resume & Interview Prep

## Resume entry
**Data Analysis AI Agent — Full Stack GenAI Application** | Python, FastAPI, Pandas, Matplotlib, LLM tool calling (Qwen/LM Studio), JavaScript | GitHub: link

- Built a full-stack GenAI app that lets users upload CSV/Excel files and query them in natural language, returning answers, tables, charts and an explanation of the method used.
- Designed a tool-calling agent in which an LLM selects from 13 validated Pandas analysis tools, eliminating arbitrary code execution and reducing hallucinated numbers.
- Developed a FastAPI REST backend with file validation, in-memory dataset sessions, rule-based fallback when the LLM is unavailable, and 35 automated pytest tests.
- Created a responsive dashboard (HTML/CSS/JS) with dataset statistics, chat with follow-up context, six chart types and fact-grounded AI insights.

(Only claim what you can explain. Edit the numbers to match your final version.)

## One query end to end (memorise this)
"Which region has the most profit?" → browser `fetch` POST `/api/ask` → FastAPI validates JSON (Pydantic) and finds the DataFrame → agent builds a system prompt with the schema → LLM returns a `group_data` tool call → `run_tool` checks the name against the whitelist, resolves column names, runs `df.groupby("Region")["Profit"].sum()` → result goes back to the LLM → LLM writes the answer → API returns answer + table + steps → browser renders them.

## Questions and short answers
**Python**
- *Why Pandas?* Fast, readable, vectorised operations on tabular data (groupby, describe, missing-value handling).
- *DataFrame?* A 2-D table with labelled columns, each column a typed Series.
- *Missing values?* Detected with `isna().sum()`, reported per column and as a percentage; I report them rather than silently filling them, so the user decides.

**FastAPI**
- *Why?* Fast, async-capable, automatic validation with Pydantic, auto docs at `/docs`.
- *API?* A defined way for two programs to talk — here, browser ↔ Python.
- *GET vs POST?* GET retrieves data and has no body; POST sends data (files/JSON) to be processed.
- *CORS?* Browser security rule restricting cross-origin requests; the server whitelists allowed origins.

**GenAI**
- *LLM?* A neural network trained on text to predict the next token; it can follow instructions.
- *Prompt engineering?* Writing clear instructions, rules and context so output is reliable (my system prompt forces tool use and bans guessing).
- *Tool calling?* The model returns structured JSON naming a function and arguments; my code executes it and returns the result.
- *Hallucination?* Confident but unsupported output. I reduce it with tools, low temperature, and fact-based insights.
- *Why Qwen?* Free to run locally, private (data never leaves the machine), supports tool calling. Trade-off: weaker than large cloud models, so I added a fallback and validation.

**Agent**
- *What makes it an agent?* It decides which tool to use, observes the result, and can chain several steps before answering.
- *How does it choose a tool?* The LLM sees each tool's name, description and parameters, plus the dataset schema.
- *Unsafe code?* It never executes model-written code. Whitelisted functions, validated columns, argument type filtering, row limits.

**Project**
- *Biggest challenge?* Example: pandas 3 changed string dtypes, which broke my categorical detection; I caught it with a smoke test and fixed it. (Use your own real challenge too.)
- *Version 2?* Persistent storage, auth, streaming, multi-step planning, Docker deploy, evaluation set for answer accuracy.
