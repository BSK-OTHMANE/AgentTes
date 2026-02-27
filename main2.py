"""
Multi-Agent Research Assistant  —  v2 (Feedback Loop)
=======================================================
Flow:
  Planner → Web Search → Summarizer ──(gaps found, iter < MAX)──▶ Planner
                              │
                         (confident OR max iterations)
                              ▼
                             END

Agents communicate via shared ResearchState:
  • Summarizer writes  →  confidence_score, gap_questions, agent_log
  • Planner reads      →  gap_questions  (treats them as the new sub-tasks)

Stack: LangGraph + DeepSeek (via OpenAI-compatible API) + Tavily
"""

import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_tavily import TavilySearch as _TavilyBackend
    _USE_NEW_TAVILY = True
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults as _TavilyBackend
    _USE_NEW_TAVILY = False

from langgraph.graph import StateGraph, END

load_dotenv()

MAX_ITERATIONS = 3          # safety cap on feedback loops
CONFIDENCE_THRESHOLD = 0.8  # summarizer must reach this to stop looping


# ─────────────────────────────────────────────────────────────
# 1.  LLM
# ─────────────────────────────────────────────────────────────
def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        temperature=temperature,
    )


# ─────────────────────────────────────────────────────────────
# 2.  Shared State  ← the "message board" agents communicate through
# ─────────────────────────────────────────────────────────────
class AgentMessage(TypedDict):
    agent: str       # who sent this message
    content: str     # what they said

class ResearchState(TypedDict):
    # core
    query:            str
    iteration:        int
    sub_tasks:        List[str]
    search_results:   List[str]    # cumulative across all iterations
    final_report:     str          # latest draft (may not be the best)

    # best result tracking
    best_report:      str          # report with highest confidence so far
    best_confidence:  float        # highest confidence score seen so far
    best_iteration:   int          # which iteration produced the best report

    # inter-agent communication
    confidence_score: float        # Summarizer -> Router
    gap_questions:    List[str]    # Summarizer -> Planner
    agent_log:        List[AgentMessage]   # full trail


# ─────────────────────────────────────────────────────────────
# 3.  Agent Nodes
# ─────────────────────────────────────────────────────────────

def _log(state: ResearchState, agent: str, msg: str) -> List[AgentMessage]:
    entry: AgentMessage = {"agent": agent, "content": msg}
    return state.get("agent_log", []) + [entry]


# ── 3a. Planner ──────────────────────────────────────────────
def planner_node(state: ResearchState) -> ResearchState:
    iteration     = state.get("iteration", 0)
    gap_questions = state.get("gap_questions", [])

    # Loop-back: Summarizer handed us gap questions — use them directly
    if iteration > 0 and gap_questions:
        print(f"\n[🧠 PLANNER]  Iteration {iteration + 1} — received {len(gap_questions)} gap(s) from Summarizer:")
        for i, q in enumerate(gap_questions, 1):
            print(f"      ↩  {i}. {q}")

        log = _log(state, "Planner",
                   f"Iteration {iteration+1}: adopting gap questions from Summarizer → {gap_questions}")
        return {**state, "sub_tasks": gap_questions, "iteration": iteration + 1, "agent_log": log}

    # First run: decompose the original query
    print(f"\n[🧠 PLANNER]  Iteration 1 — decomposing query …")
    llm = get_llm(temperature=0.2)

    system = SystemMessage(content=(
        "You are a research planner. "
        "Given a research question, output exactly 3 focused sub-questions "
        "that together cover the topic comprehensively. "
        "Return ONLY a numbered list, one sub-question per line, no extra text."
    ))
    human = HumanMessage(content=f"Research question: {state['query']}")
    raw = llm.invoke([system, human]).content.strip()

    sub_tasks = []
    for line in raw.splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            clean = line.lstrip("0123456789). ").strip()
            if clean:
                sub_tasks.append(clean)

    print(f"    Sub-tasks: {len(sub_tasks)}")
    for i, t in enumerate(sub_tasks, 1):
        print(f"      {i}. {t}")

    log = _log(state, "Planner",
               f"Iteration 1: decomposed query into {len(sub_tasks)} sub-tasks → {sub_tasks}")
    return {
        **state,
        "sub_tasks":        sub_tasks,
        "iteration":        1,
        "gap_questions":    [],
        "confidence_score": 0.0,
        "agent_log":        log,
    }


# ── 3b. Web Search ───────────────────────────────────────────
def search_node(state: ResearchState) -> ResearchState:
    iteration = state.get("iteration", 1)
    print(f"\n[🔍 WEB SEARCH]  Iteration {iteration} — searching {len(state['sub_tasks'])} question(s) …")

    search_tool = _TavilyBackend(
        max_results=3,
        tavily_api_key=os.environ["TAVILY_API_KEY"],
    )

    new_snippets: List[str] = []
    for task in state["sub_tasks"]:
        print(f"    → {task}")
        results = search_tool.invoke({"query": task})

        if isinstance(results, str):
            new_snippets.append(results)
            continue
        for r in results:
            if isinstance(r, dict):
                snippet = f"[{r.get('url', r.get('source', ''))}]\n{r.get('content', r.get('text', str(r)))}"
            else:
                snippet = str(r)
            new_snippets.append(snippet)

    # Accumulate across iterations so Summarizer always has full context
    cumulative = state.get("search_results", []) + new_snippets
    print(f"    New: {len(new_snippets)}  |  Total accumulated: {len(cumulative)}")

    log = _log(state, "WebSearch",
               f"Iteration {iteration}: +{len(new_snippets)} snippets (total: {len(cumulative)})")
    return {**state, "search_results": cumulative, "agent_log": log}


# ── 3c. Summarizer ───────────────────────────────────────────
def summarizer_node(state: ResearchState) -> ResearchState:
    iteration = state.get("iteration", 1)
    print(f"\n[📝 SUMMARIZER]  Iteration {iteration} — synthesising {len(state['search_results'])} snippets …")

    llm = get_llm(temperature=0.4)
    combined = "\n\n---\n\n".join(state["search_results"])

    system = SystemMessage(content="""You are a critical research analyst.

Your response has TWO parts:

PART 1 — Write a research report in markdown:
  Include: Title, Executive Summary, Key Findings (with source URLs), Conclusion.

PART 2 — Self-evaluate and append a JSON block at the very end:
  ```json
  {
    "confidence": <float 0.0-1.0>,
    "gaps": ["specific question still unanswered", "..."]
  }
  ```
  Rules:
  - confidence = how completely the report answers the original question (0=poor, 1=complete)
  - gaps = questions that if searched would significantly improve the report
  - If confidence >= 0.8, set gaps to []

The JSON block MUST be the last thing in your response.""")

    human = HumanMessage(content=(
        f"Original question: {state['query']}\n\n"
        f"Iteration: {iteration} of max {MAX_ITERATIONS}\n\n"
        f"Raw search results:\n{combined}"
    ))

    raw_response = llm.invoke([system, human]).content.strip()

    # ── Parse trailing JSON block ──
    confidence    = 0.5
    gap_questions: List[str] = []
    report        = raw_response

    try:
        json_start = raw_response.rfind("```json")
        json_end   = raw_response.rfind("```", json_start + 1)
        if json_start != -1 and json_end != -1:
            json_str      = raw_response[json_start + 7 : json_end].strip()
            parsed        = json.loads(json_str)
            confidence    = float(parsed.get("confidence", 0.5))
            gap_questions = [g for g in parsed.get("gaps", []) if g.strip()]
            report        = raw_response[:json_start].strip()
    except Exception:
        pass

    status = "✓ complete" if confidence >= CONFIDENCE_THRESHOLD else f"↩ needs more research"
    print(f"    Confidence: {confidence:.2f}  {status}")
    if gap_questions:
        print(f"    Gaps → sending to Planner:")
        for g in gap_questions:
            print(f"      ❓ {g}")
    else:
        print("    No gaps identified.")

    # Update best-ever tracking
    prev_best_confidence = state.get("best_confidence", 0.0)
    if confidence > prev_best_confidence:
        best_report     = report
        best_confidence = confidence
        best_iteration  = iteration
        print(f"    New best confidence: {confidence:.2f} (was {prev_best_confidence:.2f}) -> saved as best report.")
    else:
        best_report     = state.get("best_report", report)
        best_confidence = prev_best_confidence
        best_iteration  = state.get("best_iteration", iteration)
        print(f"    Previous best confidence {prev_best_confidence:.2f} still holds.")

    log = _log(state, "Summarizer",
               f"Iteration {iteration}: confidence={confidence:.2f}, best_so_far={best_confidence:.2f}, gaps={gap_questions or 'none'}")
    return {
        **state,
        "final_report":     report,
        "confidence_score": confidence,
        "gap_questions":    gap_questions,
        "best_report":      best_report,
        "best_confidence":  best_confidence,
        "best_iteration":   best_iteration,
        "agent_log":        log,
    }


# ─────────────────────────────────────────────────────────────
# 4.  Conditional Edge — the feedback decision
# ─────────────────────────────────────────────────────────────
def should_continue(state: ResearchState) -> str:
    confidence = state.get("confidence_score", 0.0)
    gaps       = state.get("gap_questions", [])
    iteration  = state.get("iteration", 1)

    if iteration >= MAX_ITERATIONS:
        print(f"\n[⚠️  ROUTER]  Max iterations ({MAX_ITERATIONS}) reached → finishing.")
        return "end"

    if confidence >= CONFIDENCE_THRESHOLD or not gaps:
        print(f"\n[✅ ROUTER]  Confidence {confidence:.2f} ≥ {CONFIDENCE_THRESHOLD} → finishing.")
        return "end"

    print(f"\n[🔁 ROUTER]  Confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD} "
          f"with {len(gaps)} gap(s) → looping back to Planner.")
    return "planner"


# ─────────────────────────────────────────────────────────────
# 5.  Build the LangGraph
# ─────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner",    planner_node)
    graph.add_node("web_search", search_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner",    "web_search")
    graph.add_edge("web_search", "summarizer")

    # The feedback loop lives here
    graph.add_conditional_edges(
        "summarizer",
        should_continue,
        {
            "planner": "planner",  # ← loop back
            "end":     END,        # ← done
        }
    )

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# 6.  CLI entry point
# ─────────────────────────────────────────────────────────────
def run(query: str) -> str:
    app = build_graph()

    initial_state: ResearchState = {
        "query":            query,
        "iteration":        0,
        "sub_tasks":        [],
        "search_results":   [],
        "final_report":     "",
        "best_report":      "",
        "best_confidence":  0.0,
        "best_iteration":   0,
        "confidence_score": 0.0,
        "gap_questions":    [],
        "agent_log":        [],
    }

    print(f"\n{'='*60}")
    print(f"  Multi-Agent Research Assistant  [v2 — Feedback Loop]")
    print(f"{'='*60}")
    print(f"  Query              : {query}")
    print(f"  Max iterations     : {MAX_ITERATIONS}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")

    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print("  AGENT COMMUNICATION LOG")
    print(f"{'='*60}")
    for entry in final_state.get("agent_log", []):
        print(f"  [{entry['agent']}]  {entry['content']}")

    print(f"\n{'='*60}")
    print(f"  FINAL REPORT  "
          f"(best from iteration {final_state['best_iteration']} of {final_state['iteration']}, "
          f"confidence: {final_state['best_confidence']:.2f})")
    print(f"{'='*60}\n")
    print(final_state["best_report"])

    return final_state["best_report"]


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What are the latest breakthroughs in quantum computing in 2024?"
    run(query)