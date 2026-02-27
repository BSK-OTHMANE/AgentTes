# Multi-Agent Research Assistant
### LangGraph + DeepSeek — POC

A progressive series of three implementations, each adding a more sophisticated orchestration pattern on top of the previous one.

---

## Project Structure

```
.
├── main.py                  # v1 — Linear Pipeline
├── main2.py         # v2 — Feedback Loop
├── main3.py       # v3 — Supervisor Pattern
├── requirements.txt
├── .env
└── README.md
```

---

## The Three Versions

### v1 — Linear Pipeline (`main.py`)

The simplest architecture. Agents run in a fixed, hardcoded sequence — no communication, no loops.

```
Planner ──► Web Search ──► Summarizer ──► END
```

| Agent | Role |
|---|---|
| Planner | Breaks the query into 3 sub-questions |
| Web Search | Runs Tavily search for each sub-question |
| Summarizer | Synthesises all snippets into a markdown report |

**When to use:** quick prototyping, simple queries, when you just need one clean pass.

---

### v2 — Feedback Loop (`main_feedback.py`)

The Summarizer evaluates its own output and writes gap questions back to the Planner if it isn't confident enough. The Planner reads those gaps and triggers a new search round.

```
Planner ──► Web Search ──► Summarizer
   ▲                           │
   │        (confidence < 0.8  │
   │         AND gaps found)   │
   └───────────────────────────┘
                           │
              (confidence >= 0.8 OR max iterations)
                           │
                          END
```

**What agents share via state:**

```python
confidence_score: float      # Summarizer → Router:  "I'm 0.6 confident"
gap_questions:    List[str]  # Summarizer → Planner: "I still need to know..."
best_report:      str        # tracks the highest-confidence report across all iterations
best_confidence:  float
```

**Key constants (top of file):**
```python
MAX_ITERATIONS = 3        # hard cap on feedback loops
CONFIDENCE_THRESHOLD = 0.8
```

**When to use:** queries where one search pass may miss context, or when report quality needs to be maximised automatically.

---

### v3 — Supervisor Pattern (`main_supervisor.py`)

A Supervisor LLM sits above all agents and dynamically decides who runs next after every single step. There are no hardcoded edges — the Supervisor reasons from the full agent log and current state.

```
                  ┌──────────────────────────────┐
                  │          SUPERVISOR           │
                  │  (LLM — decides who's next)  │
                  └──┬───────────┬───────────┬───┘
                     │           │           │
                  Planner   Web Search   Summarizer
                     │           │           │
                     └───────────┴───────────┘
                                 │
                          (all report back
                           to Supervisor)
```

The Supervisor receives a structured prompt with the full agent log, snippet count, confidence score, and gap questions, then outputs:

```json
{ "next": "web_search", "reason": "gap questions need new data before re-summarizing" }
```

This decision is stored as `next_agent` in the shared state and read by the conditional edge router.

**What this unlocks that v2 cannot do:**
- Skip the Planner and go straight to Search if sub-tasks already exist
- Call Search twice in a row if the first pass was insufficient
- Call Summarizer early if the Supervisor judges there's already enough data
- Apply different logic depending on the content of the results, not just a confidence threshold

**Key constant:**
```python
MAX_STEPS = 10   # hard cap on total agent calls
```

**When to use:** complex, open-ended queries; when you want the orchestration itself to be intelligent and adaptive rather than rule-based.

---

## Version Comparison

| | v1 Linear | v2 Feedback Loop | v3 Supervisor |
|---|---|---|---|
| Agent order | Fixed | Fixed with loop | Dynamic |
| Can skip agents? | No | No | Yes |
| Can call Search twice? | No | No | Yes |
| Routing intelligence | None | Threshold rule | LLM reasoning |
| Inter-agent comms | None | confidence + gaps | confidence + gaps + full log |
| Best report tracking | No | Yes | Yes |
| Cost | Lowest | Medium | Highest |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
```

Edit `.env`:
```
DEEPSEEK_API_KEY=your_key_here    # https://platform.deepseek.com/api_keys
TAVILY_API_KEY=your_key_here      # https://app.tavily.com  (free tier available)
```

### 3. Run any version
```bash
# v1 — simple pipeline
python main.py "What is the impact of AI on drug discovery in 2024?"

# v2 — feedback loop
python main2.py "What is the impact of AI on drug discovery in 2024?"

# v3 — supervisor
python main3.py "What is the impact of AI on drug discovery in 2024?"
```

All three accept a query as a command-line argument. If none is provided they default to a quantum computing query.

---

## How Confidence Works

Confidence is **LLM self-evaluation** — the Summarizer scores its own report on a 0–1 scale based on how completely it answers the original question. This has known limitations (overconfidence, inconsistency) and can be improved by switching to criteria-based scoring or a separate Critic agent.

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Agent graph orchestration |
| `langchain-openai` | OpenAI-compatible client (used for DeepSeek) |
| `langchain-tavily` | Web search via Tavily API |
| `langchain-community` | Fallback Tavily wrapper for older installs |
| `python-dotenv` | `.env` file loading |

