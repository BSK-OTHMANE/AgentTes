"""
Multi-Agent Research Assistant  —  v3 (Supervisor Pattern)
============================================================

Architecture:
                        ┌─────────────────────────────────┐
                        │           SUPERVISOR             │
                        │  (LLM that decides who's next)  │
                        └──┬──────────┬──────────┬────────┘
                           │          │          │
                     [planner]  [web_search]  [summarizer]
                           │          │          │
                           └──────────┴──────────┘
                                      │
                               reports back to
                                  SUPERVISOR
                                      │
                              (until SUPERVISOR
                               decides → END)

Key difference from v2:
  - No hardcoded edges like  planner → search → summarizer
  - The Supervisor reads the full agent_log and state, then decides
    which agent to call next (or to finish)
  - Agents can be called in any order, multiple times, or skipped
  - The Supervisor can ask the Planner to refine, skip straight to
    Summarizer, or call Search twice in a row if needed

Stack: LangGraph + DeepSeek + Tavily
"""

import os
import json
from typing import TypedDict, List, Literal
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

MAX_STEPS = 10   # hard cap on total agent calls to prevent infinite loops


# =============================================================
# 1.  LLM
# =============================================================
def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        temperature=temperature,
    )


# =============================================================
# 2.  Shared State
# =============================================================
class AgentMessage(TypedDict):
    agent:   str
    content: str

class ResearchState(TypedDict):
    # core
    query:            str
    step:             int            # total agent calls so far
    sub_tasks:        List[str]      # latest questions to search
    search_results:   List[str]      # cumulative snippets
    final_report:     str            # latest report draft

    # best result tracking
    best_report:      str
    best_confidence:  float
    best_step:        int

    # inter-agent communication
    confidence_score: float
    gap_questions:    List[str]
    supervisor_note:  str            # Supervisor's reasoning (visible to all agents)
    next_agent:       str            # Supervisor's routing decision (persisted in state)
    agent_log:        List[AgentMessage]


# =============================================================
# 3.  Helper
# =============================================================
def _log(state: ResearchState, agent: str, msg: str) -> List[AgentMessage]:
    entry: AgentMessage = {"agent": agent, "content": msg}
    return state.get("agent_log", []) + [entry]

def _divider(label: str):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")


# =============================================================
# 4.  Supervisor Node  ← the brain of the system
# =============================================================
def supervisor_node(state: ResearchState) -> ResearchState:
    step = state.get("step", 0)
    _divider(f"[👔 SUPERVISOR]  Step {step} — deciding next action …")

    # Build a concise status summary for the Supervisor LLM
    log_summary = "\n".join(
        f"  [{e['agent']}] {e['content']}"
        for e in state.get("agent_log", [])
    ) or "  (no actions taken yet)"

    system = SystemMessage(content="""You are the Supervisor of a multi-agent research system.
You coordinate three specialist agents to answer a research question:

  - planner     : breaks the query (or gap questions) into focused sub-questions to search
  - web_search  : searches the web for the current sub_tasks
  - summarizer  : synthesises all search results into a report and scores confidence (0-1)

Your job: read the current state and decide which agent to call next, or whether to finish.

Routing rules:
  - Start with 'planner' if no sub_tasks exist yet.
  - Always call 'web_search' after 'planner' so there is something to summarise.
  - Call 'summarizer' once enough results are collected.
  - If confidence < 0.75 and there are gap_questions, call 'planner' to address the gaps,
    then 'web_search', then 'summarizer' again.
  - If confidence >= 0.75, or no meaningful gaps remain, or steps are running high → 'FINISH'.

Respond with ONLY a JSON object, nothing else:
{
  "next": "<planner | web_search | summarizer | FINISH>",
  "reason": "<one sentence explaining your decision>"
}""")

    human = HumanMessage(content=f"""Research question: {state['query']}

Current step: {step} / max {MAX_STEPS}
Sub-tasks: {state.get('sub_tasks', [])}
Search snippets collected: {len(state.get('search_results', []))}
Latest confidence score: {state.get('confidence_score', 0.0):.2f}
Best confidence so far: {state.get('best_confidence', 0.0):.2f}
Gap questions from Summarizer: {state.get('gap_questions', [])}
Supervisor previous note: {state.get('supervisor_note', 'none')}

Agent log:
{log_summary}

What should happen next?""")

    raw = get_llm(temperature=0.0).invoke([system, human]).content.strip()

    # Parse JSON
    next_agent = "planner"
    reason = ""
    try:
        # Strip markdown fences if present
        clean = raw
        if "```" in clean:
            clean = clean[clean.find("{"):clean.rfind("}")+1]
        parsed     = json.loads(clean)
        next_agent = parsed.get("next", "planner").strip()
        reason     = parsed.get("reason", "")
    except Exception:
        # Fallback: scan for keywords
        for keyword in ["FINISH", "summarizer", "web_search", "planner"]:
            if keyword.lower() in raw.lower():
                next_agent = keyword
                break

    print(f"  Decision : {next_agent}")
    print(f"  Reason   : {reason}")

    log = _log(state, "Supervisor", f"Step {step}: route → {next_agent} | {reason}")

    return {
        **state,
        "step":            step + 1,       # increment here so agents see accurate count
        "supervisor_note": reason,
        "next_agent":      next_agent,     # properly persisted in state
        "agent_log":       log,
    }


# =============================================================
# 5.  Agent Nodes
# =============================================================

# ── 5a. Planner ──────────────────────────────────────────────
def planner_node(state: ResearchState) -> ResearchState:
    step          = state.get("step", 0)
    gap_questions = state.get("gap_questions", [])

    _divider(f"[🧠 PLANNER]  Step {step}")

    llm = get_llm(temperature=0.2)

    # If Supervisor sent us gap questions, use them; else decompose original query
    if gap_questions:
        print(f"  Addressing {len(gap_questions)} gap question(s) from Summarizer:")
        for i, q in enumerate(gap_questions, 1):
            print(f"    {i}. {q}")
        sub_tasks = gap_questions
        log_msg   = f"Step {step}: adopted {len(gap_questions)} gap questions as sub-tasks."
    else:
        print(f"  Decomposing original query …")
        system = SystemMessage(content=(
            "You are a research planner. "
            "Break the research question into exactly 3 focused sub-questions "
            "that together cover the topic. "
            "Return ONLY a numbered list, one sub-question per line."
        ))
        human  = HumanMessage(content=f"Research question: {state['query']}")
        raw    = llm.invoke([system, human]).content.strip()

        sub_tasks = []
        for line in raw.splitlines():
            line = line.strip()
            if line and line[0].isdigit():
                clean = line.lstrip("0123456789). ").strip()
                if clean:
                    sub_tasks.append(clean)

        log_msg = f"Step {step}: decomposed query into {len(sub_tasks)} sub-tasks."

    print(f"  Sub-tasks ({len(sub_tasks)}):")
    for i, t in enumerate(sub_tasks, 1):
        print(f"    {i}. {t}")

    log = _log(state, "Planner", log_msg)
    return {**state, "sub_tasks": sub_tasks, "gap_questions": [], "step": step, "agent_log": log}


# ── 5b. Web Search ───────────────────────────────────────────
def search_node(state: ResearchState) -> ResearchState:
    step = state.get("step", 0)
    _divider(f"[🔍 WEB SEARCH]  Step {step} — {len(state.get('sub_tasks', []))} question(s)")

    search_tool = _TavilyBackend(
        max_results=3,
        tavily_api_key=os.environ["TAVILY_API_KEY"],
    )

    new_snippets: List[str] = []
    for task in state.get("sub_tasks", []):
        print(f"  → {task}")
        results = search_tool.invoke({"query": task})
        # NEW Tavily backend returns dict
        if isinstance(results, dict):
            results_list = results.get("results", [])
        else:
            results_list = results

        for r in results_list:
            if isinstance(r, dict):
                snippet = f"[{r.get('url','')}]\n{r.get('content','')}"
            else:
                snippet = str(r)

            new_snippets.append(snippet)

    cumulative = state.get("search_results", []) + new_snippets
    print(f"  New: {len(new_snippets)} snippets  |  Total: {len(cumulative)}")

    log = _log(state, "WebSearch",
               f"Step {step}: +{len(new_snippets)} snippets (total: {len(cumulative)})")
    return {**state, "search_results": cumulative, "step": step, "agent_log": log}


# ── 5c. Summarizer ───────────────────────────────────────────
def summarizer_node(state: ResearchState) -> ResearchState:
    step = state.get("step", 0)
    _divider(f"[📝 SUMMARIZER]  Step {step} — {len(state.get('search_results', []))} snippets")

    llm     = get_llm(temperature=0.4)
    combined = "\n\n---\n\n".join(state.get("search_results", []))

    system = SystemMessage(content="""You are a critical research analyst.

PART 1 — Write a research report in markdown:
  Title, Executive Summary, Key Findings (with source URLs), Conclusion.

PART 2 — Self-evaluate. Append this JSON block at the very end:
  ```json
  {
    "confidence": <float 0.0-1.0>,
    "gaps": ["question still unanswered", "..."]
  }
  ```
  - confidence: how completely the report answers the original question
  - gaps: specific questions that would significantly improve the report
  - If confidence >= 0.75, set gaps to []

The JSON block MUST be the last thing in your response.""")

    human = HumanMessage(content=(
        f"Original question: {state['query']}\n\n"
        f"Supervisor note: {state.get('supervisor_note', '')}\n\n"
        f"Raw search results:\n{combined}"
    ))

    raw_response = get_llm(temperature=0.4).invoke([system, human]).content.strip()

    # Parse trailing JSON
    confidence    = 0.5
    gap_questions: List[str] = []
    report        = raw_response
    try:
        js = raw_response.rfind("```json")
        je = raw_response.rfind("```", js + 1)
        if js != -1 and je != -1:
            parsed        = json.loads(raw_response[js + 7 : je].strip())
            confidence    = float(parsed.get("confidence", 0.5))
            gap_questions = [g for g in parsed.get("gaps", []) if g.strip()]
            report        = raw_response[:js].strip()
    except Exception:
        pass

    # Update best tracking
    prev_best = state.get("best_confidence", 0.0)
    if confidence > prev_best:
        best_report, best_confidence, best_step = report, confidence, step
        print(f"  New best confidence: {confidence:.2f} (was {prev_best:.2f}) -> saved.")
    else:
        best_report     = state.get("best_report", report)
        best_confidence = prev_best
        best_step       = state.get("best_step", step)
        print(f"  Confidence: {confidence:.2f}  (best still {prev_best:.2f} from step {best_step})")

    if gap_questions:
        print(f"  Gaps -> Supervisor will decide if we re-search:")
        for g in gap_questions:
            print(f"    ? {g}")
    else:
        print("  No gaps identified.")

    log = _log(state, "Summarizer",
               f"Step {step}: confidence={confidence:.2f}, best={best_confidence:.2f}, gaps={gap_questions or 'none'}")
    return {
        **state,
        "final_report":     report,
        "confidence_score": confidence,
        "gap_questions":    gap_questions,
        "best_report":      best_report,
        "best_confidence":  best_confidence,
        "best_step":        best_step,
        "step":             step,
        "agent_log":        log,
    }


# =============================================================
# 6.  Routing — reads Supervisor's decision from state
# =============================================================
def route_from_supervisor(state: ResearchState) -> str:
    decision = state.get("next_agent", "planner")
    step     = state.get("step", 0)

    if step >= MAX_STEPS:
        print(f"\n[⚠️  HARD STOP]  {MAX_STEPS} steps reached -> finishing.")
        return "end"

    mapping = {
        "planner":    "planner",
        "web_search": "web_search",
        "summarizer": "summarizer",
        "finish":     "end",
        "FINISH":     "end",
    }
    return mapping.get(decision, "end")


# =============================================================
# 7.  Build the LangGraph
# =============================================================
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor",  supervisor_node)
    graph.add_node("planner",     planner_node)
    graph.add_node("web_search",  search_node)
    graph.add_node("summarizer",  summarizer_node)

    # Entry point is always the Supervisor
    graph.set_entry_point("supervisor")

    # After every agent, report back to Supervisor
    for agent in ["planner", "web_search", "summarizer"]:
        graph.add_edge(agent, "supervisor")

    # Supervisor decides who's next (or END)
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "planner":    "planner",
            "web_search": "web_search",
            "summarizer": "summarizer",
            "end":        END,
        }
    )

    return graph.compile()


# =============================================================
# 8.  CLI entry point
# =============================================================
def run(query: str) -> str:
    app = build_graph()

    initial_state: ResearchState = {
        "query":            query,
        "step":             0,
        "sub_tasks":        [],
        "search_results":   [],
        "final_report":     "",
        "best_report":      "",
        "best_confidence":  0.0,
        "best_step":        0,
        "confidence_score": 0.0,
        "gap_questions":    [],
        "supervisor_note":  "",
        "next_agent":       "",
        "agent_log":        [],
    }

    print(f"\n{'='*60}")
    print(f"  Multi-Agent Research Assistant  [v3 - Supervisor]")
    print(f"{'='*60}")
    print(f"  Query     : {query}")
    print(f"  Max steps : {MAX_STEPS}")

    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print("  AGENT COMMUNICATION LOG")
    print(f"{'='*60}")
    for entry in final_state.get("agent_log", []):
        icon = {"Supervisor": "👔", "Planner": "🧠", "WebSearch": "🔍", "Summarizer": "📝"}.get(entry["agent"], "•")
        print(f"  {icon} [{entry['agent']}]  {entry['content']}")

    print(f"\n{'='*60}")
    print(f"  FINAL REPORT")
    print(f"  Best from step {final_state['best_step']} of {final_state['step']}  |  "
          f"Confidence: {final_state['best_confidence']:.2f}")
    print(f"{'='*60}\n")
    print(final_state["best_report"])

    return final_state["best_report"]


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "give me the latest update on the iran usa war"
    run(query)