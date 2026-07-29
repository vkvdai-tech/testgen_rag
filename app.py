import os
import json
import re
import streamlit as st
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from duckduckgo_search import DDGS  # Web search integration

st.set_page_config(page_title="UPSC Polity MCQ Generator", page_icon="📚", layout="centered")

st.title("📚 UPSC Polity MCQ Generator")
st.caption("Powered by ChromaDB, Web Search & OpenAI (gpt-5.6-luna)")

# API Key handling
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("Please configure OPENAI_API_KEY environment variable or Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
PRIMARY_MODEL = "gpt-5.6-luna"

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="polity_collection"
    )

vector_db = load_vector_db()

# Helper function to fetch web results
def fetch_web_context(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"UPSC Polity {query}", max_results=max_results))
            if not results:
                return ""
            formatted = [f"[Web Source: {r.get('title')}]\n{r.get('body')}" for r in results]
            return "\n\n".join(formatted)
    except Exception as e:
        st.warning(f"Web search unavailable: {e}")
        return ""

# Inputs
topic = st.text_input("Enter Primary Topic:", "Right to Equality")

enable_web_search = st.checkbox(
    "🌐 Enable Live Web Search (supplement vector DB context)", 
    value=True,
    help="When enabled, fetches live web context alongside local ChromaDB context."
)

custom_instructions = st.text_area(
    "Custom Focus or Interlinking Instructions (Optional):",
    placeholder="e.g., Interlink this topic with judicial review, Article 21 cases, or relevant constitutional amendments."
)

# Session state setup
if "max_estimate" not in st.session_state:
    st.session_state.max_estimate = None
if "selected_count" not in st.session_state or st.session_state.selected_count < 1:
    st.session_state.selected_count = 10
if "analyzed_topic" not in st.session_state:
    st.session_state.analyzed_topic = None

# Step 1: Capacity Estimation
if st.button("🔍 Estimate Max Question Capacity"):
    with st.spinner("Retrieving context & analyzing capacity with gpt-5.6-luna..."):
        # 1. Local DB Search
        db_results = vector_db.similarity_search_with_score(topic, k=8)
        db_blocks = [
            f"[DB Source: {os.path.basename(doc.metadata.get('source', 'Doc'))}]\n{doc.page_content.strip()}"
            for doc, score in db_results if doc.page_content.strip()
        ]
        
        # 2. Web Search (if checked or if DB results are thin)
        web_blocks = ""
        if enable_web_search or len(db_blocks) < 2:
            web_blocks = fetch_web_context(topic)

        combined_context = "\n\n---\n\n".join(db_blocks)
        if web_blocks:
            combined_context += "\n\n--- WEB RESULTS ---\n\n" + web_blocks

        estimate_prompt = f"""
You are an expert UPSC Civil Services examination paper setter. 
Estimate the MAXIMUM total number of distinct, high-quality UPSC-level MCQs that can be created on the topic "{topic}".

Consider both the supplied reference context and standard UPSC syllabus depth for this topic.

--- REFERENCE CONTEXT ---
{combined_context}
------------------------

Return a single valid JSON object strictly in this format:
{{"estimated_max": 50, "reason": "Brief summary of available topic depth"}}
"""
        try:
            res = client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant that outputs strictly valid JSON."},
                    {"role": "user", "content": estimate_prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw = res.choices[0].message.content.strip()
            cleaned_raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(cleaned_raw)
            
            st.session_state.max_estimate = max(5, int(data.get("estimated_max", 50)))
            st.session_state.reason = data.get("reason", "Topic depth analyzed successfully.")
            st.session_state.analyzed_topic = topic
            st.session_state.selected_count = st.session_state.max_estimate
        except Exception as e:
            st.error(f"❌ `{PRIMARY_MODEL}` API / Parse Error: {e}")

# Preset buttons & generation pipeline remain unchanged...
