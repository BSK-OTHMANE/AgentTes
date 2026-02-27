# 🔬 Multi-Agent Research Assistant
### LangGraph + DeepSeek POC

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   LangGraph DAG                      │
│                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌────────┐ │
│  │   PLANNER    │───▶│  WEB SEARCH  │───▶│SUMMARY │ │
│  │              │    │   AGENT      │    │ AGENT  │ │
│  │ Breaks query │    │              │    │        │ │
│  │ into 3 sub-  │    │ Tavily API   │    │DeepSeek│ │
│  │ tasks        │    │ per sub-task │    │report  │ │
│  └──────────────┘    └──────────────┘    └────────┘ │
│         DeepSeek            ↑                  ↑    │
│                       (3 searches)       (synthesis) │
└─────────────────────────────────────────────────────┘
    │
    ▼
Final Markdown Report
```

### Agent Roles

| Agent | Responsibility | LLM Used |
|-------|---------------|----------|
| **Planner** | Decomposes the user query into 3 focused sub-questions | DeepSeek |
| **Web Search** | Runs Tavily search for each sub-question, collects raw snippets | Tavily API |
| **Summarizer** | Synthesises all snippets into a structured markdown report | DeepSeek |

---

## Setup

### 1. Clone / copy the project
```bash
mkdir research-assistant && cd research-assistant
# copy the files here
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
```bash
cp .env.example .env
# Edit .env and add your keys:
#   DEEPSEEK_API_KEY  →  https://platform.deepseek.com/api_keys
#   TAVILY_API_KEY    →  https://app.tavily.com  (free tier available)
```

### 4. Run it
```bash
# Default query (quantum computing)
python main.py

# Custom query
python main.py "What is the impact of AI on drug discovery in 2024?"
```

---

## Sample Output

```
============================================================
  Multi-Agent Research Assistant
============================================================
  Query: What are the latest breakthroughs in quantum computing in 2024?

[🧠 PLANNER]  Decomposing query …
    Sub-tasks identified: 3
      1. What major quantum computing milestones were achieved in 2024?
      2. Which companies are leading quantum computing research in 2024?
      3. What practical applications of quantum computing emerged in 2024?

[🔍 WEB SEARCH]  Searching …
    Searching: What major quantum computing milestones were achieved in 2024?
    Searching: Which companies are leading quantum computing research in 2024?
    Searching: What practical applications of quantum computing emerged in 2024?
    Total snippets collected: 9

[📝 SUMMARIZER]  Writing report …
    Report generated ✓

============================================================
  FINAL REPORT
============================================================

# Quantum Computing Breakthroughs in 2024

## Executive Summary
...
```

---

## Extending the POC

- **Add a Critic Agent** — add a 4th node that reviews the report and requests re-search if gaps are found (conditional edges in LangGraph)
- **Persist state** — use LangGraph's checkpointer (`SqliteSaver`) for resumable runs
- **Streaming** — call `app.stream(initial_state)` instead of `app.invoke()` for token-by-token output
- **Tool calling** — give agents additional tools (calculator, code executor, PDF reader)

---

## Dependencies

- [`langgraph`](https://github.com/langchain-ai/langgraph) — agent graph orchestration
- [`langchain-openai`](https://github.com/langchain-ai/langchain) — OpenAI-compatible LLM client (used for DeepSeek)
- [`tavily-python`](https://docs.tavily.com) — web search API
- [`python-dotenv`](https://github.com/theskumar/python-dotenv) — env var management
