import streamlit as st
import importlib

# Import your 3 versions
import main       # v1
import main2      # v2
import main3      # v3


st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Research Assistant")
st.markdown("Compare different orchestration architectures (v1 / v2 / v3)")

# -------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------
st.sidebar.header("⚙️ Configuration")

version = st.sidebar.selectbox(
    "Choose Architecture",
    [
        "v1 — Linear Pipeline",
        "v2 — Feedback Loop",
        "v3 — Supervisor Pattern"
    ]
)

query = st.text_area(
    "Enter your research question:",
    "What are the latest breakthroughs in quantum computing in 2024?",
    height=120
)

run_button = st.button("🚀 Run Research")

# -------------------------------------------------
# Execution
# -------------------------------------------------
if run_button and query.strip():

    if version.startswith("v1"):
        module = main
    elif version.startswith("v2"):
        module = main2
    else:
        module = main3

    with st.spinner("Running multi-agent system... this may take a moment ⏳"):
        result = module.run(query)

    st.success("Research completed!")

    # -------------------------------------------------
    # Display Results
    # -------------------------------------------------
    st.divider()

    st.subheader("📄 Final Report")
    st.markdown(result)

    # Try to access additional state info for v2 / v3
    if version.startswith("v2") or version.startswith("v3"):
        try:
            # Re-run to capture full state
            app_graph = module.build_graph()
            final_state = app_graph.invoke({
                "query": query,
                "iteration": 0,
                "step": 0,
                "sub_tasks": [],
                "search_results": [],
                "final_report": "",
                "best_report": "",
                "best_confidence": 0.0,
                "best_iteration": 0,
                "best_step": 0,
                "confidence_score": 0.0,
                "gap_questions": [],
                "supervisor_note": "",
                "next_agent": "",
                "agent_log": [],
            })

            if "best_confidence" in final_state:
                st.metric("Confidence Score", f"{final_state['best_confidence']:.2f}")

            if "agent_log" in final_state:
                st.subheader("🧠 Agent Communication Log")
                for entry in final_state["agent_log"]:
                    st.markdown(f"**[{entry['agent']}]** — {entry['content']}")

        except Exception:
            pass