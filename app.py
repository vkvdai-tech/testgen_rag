import os
import json
import re
import streamlit as st
from openai import OpenAI
import anthropic
from google import genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

st.set_page_config(page_title="UPSC Polity MCQ Generator (V36 Elite)", page_icon="📚", layout="wide")

# --- RESET SESSION FUNCTION ---
def reset_session():
    keys_to_clear = [
        "max_estimate", "reason", "selected_count", 
        "analyzed_topic", "last_context", "generated_paper", 
        "generated_questions_history"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- SIDEBAR CONTROLS & PROMINENT MODEL SELECTOR ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # 🎯 THIS IS THE MODEL SELECTOR DROPDOWN
    selected_provider = st.selectbox(
        "🧠 Select AI Model Engine:",
        [
            "OpenAI (gpt-5.6-luna)", 
            "Claude (claude-3-5-sonnet)", 
            "Gemini (gemini-2.5-pro)"
        ],
        index=0,
        help="Choose which LLM provider generates your MCQs."
    )
    
    st.info(f"Active Model: **{selected_provider}**")
    st.markdown("---")
    
    st.header("🛠️ Session Controls")
    st.markdown("Clear session history and reset deduplication state for new topics.")
    if st.button("🔄 Reset / Clear Session", use_container_width=True):
        reset_session()

# MAIN HEADER
st.title("📚 UPSC Polity MCQ Generator")
st.caption(f"Powered by Constitution Bare Act, Indian Kanoon, Web Search & {selected_provider}")

# --- API CLIENT INITIALIZATION ---
openai_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

openai_client = OpenAI(api_key=openai_key) if openai_key else None
claude_client = anthropic.Anthropic(api_key=anthropic_key) if anthropic_key else None
gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None

def call_llm(provider: str, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    """Unified LLM router for OpenAI, Claude, and Gemini."""
    if "OpenAI" in provider:
        if not openai_client:
            raise ValueError("OPENAI_API_KEY is missing! Check .streamlit/secrets.toml or Cloud Secrets.")
        kwargs = {
            "model": "gpt-5.6-luna",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        res = openai_client.chat.completions.create(**kwargs)
        return res.choices[0].message.content.strip()

    elif "Claude" in provider:
        if not claude_client:
            raise ValueError("ANTHROPIC_API_KEY is missing! Check .streamlit/secrets.toml or Cloud Secrets.")
        res = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return res.content[0].text.strip()

    elif "Gemini" in provider:
        if not gemini_client:
            raise ValueError("GEMINI_API_KEY is missing! Check .streamlit/secrets.toml or Cloud Secrets.")
        
        prompt = f"{system_prompt}\n\n{user_prompt}"
        res = gemini_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
        )
        return res.text.strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="polity_collection"
    )

vector_db = load_vector_db()

def fetch_web_context(query: str, max_results: int = 3) -> str:
    if not HAS_DDG:
        return ""
    try:
        results = DDGS().text(keywords=f"UPSC Polity {query} PRS Legislative Research", max_results=max_results)
        if not results:
            return ""
        formatted = [f"[Web/PRS Source: {r.get('title', 'Web')}]\n{r.get('body', '')}" for r in results]
        return "\n\n".join(formatted)
    except Exception:
        return ""

def fetch_kanoon_context(query: str, max_results: int = 3) -> str:
    if not HAS_DDG:
        return ""
    try:
        results = DDGS().text(keywords=f"site:indiankanoon.org Supreme Court {query} landmark judgment ratio", max_results=max_results)
        if not results:
            return ""
        formatted = [f"[Indian Kanoon SC Precedent: {r.get('title', 'Kanoon')}]\n{r.get('body', '')}" for r in results]
        return "\n\n".join(formatted)
    except Exception:
        return ""

# MAIN TOPIC INPUT
topic = st.text_input("Enter Micro-Topic / Primary Topic:", "Anti-Defection Law")

col_a, col_b = st.columns(2)
with col_a:
    enable_kanoon = st.checkbox("⚖️ Fetch Indian Kanoon Precedents", value=True)
with col_b:
    enable_web_search = st.checkbox("🌐 Enable PRS / Live Web Search", value=True)

custom_instructions = st.text_area(
    "Custom Focus or Interlinking Instructions (Optional):",
    placeholder="e.g., Interlink this topic with judicial review, Article 102/191, landmark 10th schedule cases."
)

if "max_estimate" not in st.session_state:
    st.session_state.max_estimate = None
if "reason" not in st.session_state:
    st.session_state.reason = ""
if "selected_count" not in st.session_state or st.session_state.selected_count < 1:
    st.session_state.selected_count = 10
if "analyzed_topic" not in st.session_state:
    st.session_state.analyzed_topic = None
if "generated_questions_history" not in st.session_state:
    st.session_state.generated_questions_history = []

btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    estimate_clicked = st.button("🔍 Estimate Max Question Capacity", use_container_width=True)
with btn_col2:
    if st.button("🔄 Reset", use_container_width=True):
        reset_session()

if estimate_clicked:
    with st.spinner(f"Analyzing topic depth using {selected_provider}..."):
        db_blocks = []
        try:
            db_results = vector_db.similarity_search_with_score(topic, k=8)
            db_blocks = [
                f"[Bare Act Source: {os.path.basename(doc.metadata.get('source', 'Doc'))}]\n{doc.page_content.strip()}"
                for doc, score in db_results if doc.page_content.strip()
            ]
        except Exception:
            pass
        
        kanoon_blocks = fetch_kanoon_context(topic) if enable_kanoon else ""
        web_blocks = fetch_web_context(topic) if enable_web_search else ""

        combined_context = "\n\n---\n\n".join(db_blocks)
        if kanoon_blocks:
            combined_context += "\n\n--- INDIAN KANOON PRECEDENTS ---\n\n" + kanoon_blocks
        if web_blocks:
            combined_context += "\n\n--- LIVE WEB / PRS CONTEXT ---\n\n" + web_blocks

        estimate_prompt = f"""
Estimate the MAXIMUM total number of distinct, high-quality, non-repetitive UPSC-level MCQs that can be created STRICTLY on the micro-topic "{topic}".

--- MULTI-AUTHORITY CONTEXT ---
{combined_context}
------------------------------

Return a single valid JSON object strictly in this structure:
{{"estimated_max": 50, "reason": "Detailed conceptual coverage including constitutional inferences, statutory mechanics, and landmark Supreme Court judgments across 18 distinct UPSC question formats."}}
"""
        try:
            raw = call_llm(
                provider=selected_provider,
                system_prompt="You are an assistant that outputs strictly valid JSON.",
                user_prompt=estimate_prompt,
                json_mode=True
            )
            cleaned_raw = re.sub(r"^```(?:json)?\s*|\s*
