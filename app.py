import json
import time
import streamlit as st
from src.agents.orchestrator_agent import parse_user_query
from src.tools.search_tools import hybrid_search
from src.agents.presenter_agent import format_itinerary

st.set_page_config(
    page_title="Indian Travel Planner",
    page_icon="🧭",
    layout="wide",
)

EXAMPLES = [
    "5 day budget trip to Goa with friends, beaches and seafood",
    "Peaceful 3 day mountain getaway in Himachal for a family with kids",
    "Luxury 7 day heritage trip in Rajasthan",
    "Weekend offbeat trip somewhere in South India under a small budget",
]

def run_pipeline(user_prompt: str, top_k: int, status):
    """Runs the 3-step agent pipeline, updating the Streamlit status widget."""
    timings = {}

    status.update(label="[1/3] Parsing query with Orchestrator Agent...")
    t0 = time.perf_counter()
    params = parse_user_query(user_prompt)
    timings["Orchestrator"] = time.perf_counter() - t0

    status.update(label="[2/3] Querying SQLite and ChromaDB vector store...")
    t0 = time.perf_counter()
    results = hybrid_search(params, top_k=top_k)
    timings["Hybrid search"] = time.perf_counter() - t0

    status.update(label="[3/3] Formulating itinerary with Presenter Agent...")
    t0 = time.perf_counter()
    itinerary = format_itinerary(user_prompt, results, params)
    timings["Presenter"] = time.perf_counter() - t0

    return params, results, itinerary, timings

def render_params(params: dict):
    """Show the constraints the orchestrator extracted from the query."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Destination", params.get("destination_name") or "Flexible")
    duration = params.get("duration_days")
    c2.metric("Duration", f"{duration} days" if duration else "Not specified")
    c3.metric("Budget tier", str(params.get("budget_tier") or "Not specified").title())

    with st.expander("All extracted parameters"):
        st.json(params)

def render_results(results):
    """Show raw matching records. Structure is unknown, so render defensively."""
    if not results:
        st.info("No matching destination records found.")
        return

    for i, rec in enumerate(results, start=1):
        if isinstance(rec, dict):
            title = (
                rec.get("name")
                or rec.get("destination_name")
                or rec.get("title")
                or f"Match {i}"
            )
            with st.expander(f"{i}. {title}", expanded=(i == 1)):
                st.json(rec)
        else:
            with st.expander(f"Match {i}", expanded=(i == 1)):
                try:
                    st.json(json.loads(json.dumps(rec, default=str)))
                except Exception:
                    st.write(rec)

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider(
        "Destinations to retrieve (top_k)",
        min_value=1,
        max_value=5,
        value=2,
        help="How many matching destination records the hybrid search returns.",
    )
    show_debug = st.checkbox("Show retrieved records and timings", value=True)

    st.divider()
    st.subheader("💡 Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state["query_input"] = ex

    st.divider()
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("🧭 Offline Indian Travel Planner")
st.caption("Describe your trip in plain language. Everything runs on your local database and vector store.")

with st.form("query_form", clear_on_submit=False):
    query = st.text_area(
        "Your travel request",
        key="query_input",
        height=110,
        placeholder="e.g. 5 day budget trip to Goa with friends, beaches and seafood",
    )
    submitted = st.form_submit_button("Plan my trip ✈️", type="primary")

if submitted:
    if not query.strip():
        st.warning("Please enter a travel request first.")
    else:
        try:
            with st.status("Starting...", expanded=True) as status:
                params, results, itinerary, timings = run_pipeline(query.strip(), top_k, status)
                status.update(label="Done!", state="complete", expanded=False)

            st.session_state["history"].insert(
                0,
                {
                    "query": query.strip(),
                    "params": params,
                    "results": results,
                    "itinerary": itinerary,
                    "timings": timings,
                },
            )
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
            st.exception(e)

for idx, item in enumerate(st.session_state["history"]):
    is_latest = idx == 0
    label = f"✈️ {item['query']}"

    with st.expander(label, expanded=is_latest):
        st.subheader("Understood constraints")
        render_params(item["params"])

        st.subheader("Your itinerary")
        st.markdown(item["itinerary"])

        st.download_button(
            "⬇️ Download itinerary (.md)",
            data=str(item["itinerary"]),
            file_name="itinerary.md",
            mime="text/markdown",
            key=f"dl_{idx}",
        )

        if show_debug:
            st.divider()
            st.subheader("Retrieved destination records")
            render_results(item["results"])

            st.caption(
                " · ".join(f"{k}: {v:.2f}s" for k, v in item["timings"].items())
            )
