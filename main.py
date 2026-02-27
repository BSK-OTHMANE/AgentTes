"""
Multi-Agent Research Assistant
================================
Agents:
  1. Planner / Orchestrator  — breaks the query into sub-tasks
  2. Web Search Agent        — fetches results via Tavily
  3. Summarizer Agent        — distils raw results into a final report

Stack: LangGraph + DeepSeek (via OpenAI-compatible API)
"""

import os
from typing import TypedDict, Annotated, List
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

# ─────────────────────────────────────────────
# 1.  LLM  (DeepSeek via OpenAI-compatible API)
# ─────────────────────────────────────────────
def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        temperature=temperature,
    )


# ─────────────────────────────────────────────
# 2.  Shared State
# ─────────────────────────────────────────────
class ResearchState(TypedDict):
    query: str                   # original user query
    sub_tasks: List[str]         # sub-questions from the planner
    search_results: List[str]    # raw snippets from web search
    final_report: str            # polished summary


# ─────────────────────────────────────────────
# 3.  Agent Nodes
# ─────────────────────────────────────────────

def planner_node(state: ResearchState) -> ResearchState:
    """Breaks the user query into focused sub-tasks."""
    print("\n[🧠 PLANNER]  Decomposing query …")
    llm = get_llm(temperature=0.2)

    system = SystemMessage(content=(
        "You are a research planner. "
        "Given a research question, output 3 focused sub-questions "
        "that together cover the topic comprehensively. "
        "Return ONLY a numbered list, one sub-question per line."
    ))
    human = HumanMessage(content=f"Research question: {state['query']}")

    response = llm.invoke([system, human])
    raw = response.content.strip()

    # Parse numbered list into clean strings
    sub_tasks = []
    for line in raw.splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            # strip leading "1. " etc.
            clean = line.lstrip("0123456789. ").strip()
            if clean:
                sub_tasks.append(clean)

    print(f"    Sub-tasks identified: {len(sub_tasks)}")
    for i, t in enumerate(sub_tasks, 1):
        print(f"      {i}. {t}")

    return {**state, "sub_tasks": sub_tasks}


def search_node(state: ResearchState) -> ResearchState:
    """Runs a Tavily search for each sub-task and collects snippets."""
    print("\n[🔍 WEB SEARCH]  Searching …")

    if _USE_NEW_TAVILY:
        # langchain-tavily >= 0.1  →  TavilySearch
        search_tool = _TavilyBackend(
            max_results=3,
            tavily_api_key=os.environ["TAVILY_API_KEY"],
        )
    else:
        # langchain-community (legacy)
        search_tool = _TavilyBackend(
            max_results=3,
            tavily_api_key=os.environ["TAVILY_API_KEY"],
        )

    all_results: List[str] = []
    for task in state["sub_tasks"]:
        print(f"    Searching: {task}")
        results = search_tool.invoke({"query": task})

        # Normalise: results can be a list of dicts OR a single string
        if isinstance(results, str):
            all_results.append(results)
            continue

        for r in results:
            if isinstance(r, dict):
                snippet = f"[{r.get('url', r.get('source', ''))}]\n{r.get('content', r.get('text', str(r)))}"
            else:
                snippet = str(r)
            all_results.append(snippet)

    print(f"    Total snippets collected: {len(all_results)}")
    return {**state, "search_results": all_results}


def summarizer_node(state: ResearchState) -> ResearchState:
    """Synthesises search snippets into a structured final report."""
    print("\n[📝 SUMMARIZER]  Writing report …")
    llm = get_llm(temperature=0.4)

    combined = "\n\n---\n\n".join(state["search_results"])

    system = SystemMessage(content=(
        "You are a research analyst. "
        "Given raw web search snippets, write a clear, well-structured research report "
        "that directly answers the original question. "
        "Use markdown: include a title, short executive summary, "
        "key findings (with source URLs), and a conclusion. "
        "Be concise but comprehensive."
    ))
    human = HumanMessage(content=(
        f"Original question: {state['query']}\n\n"
        f"Raw search results:\n{combined}"
    ))

    response = llm.invoke([system, human])
    report = response.content.strip()

    print("    Report generated ✓")
    return {**state, "final_report": report}


# ─────────────────────────────────────────────
# 4.  Build the LangGraph
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("planner",    planner_node)
    graph.add_node("web_search", search_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner",    "web_search")
    graph.add_edge("web_search", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 5.  CLI entry point
# ─────────────────────────────────────────────

def run(query: str) -> str:
    app = build_graph()

    initial_state: ResearchState = {
        "query":          query,
        "sub_tasks":      [],
        "search_results": [],
        "final_report":   "",
    }

    print(f"\n{'='*60}")
    print(f"  Multi-Agent Research Assistant")
    print(f"{'='*60}")
    print(f"  Query: {query}")

    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print("  FINAL REPORT")
    print(f"{'='*60}\n")
    print(final_state["final_report"])

    return final_state["final_report"]


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What are the latest breakthroughs in quantum computing in 2024?"
    run(query)