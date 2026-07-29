import os
import json
import re
import streamlit as st
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Safe import for duckduckgo-search
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

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

# Safe Web Search Helper
def fetch_web_context(query: str, max_results: int = 5) -> str:
    if not HAS_DDG:
        return ""
    try:
        results = DDGS().text(keywords=f"UPSC Polity {query}", max_results=max_results)
        if not results:
            return ""
        formatted = [f"[Web Source: {r.get('title', 'Web')}]\n{r.get('body', '')}" for r in results]
        return "\n\n".join(formatted)
    except Exception as e:
        st.warning(f"⚠️ Live web retrieval paused ({e}). Continuing with LLM evaluation.")
        return ""

# Inputs
topic = st.text_input("Enter Primary Topic:", "Anti-Defection Law")

enable_web_search = st.checkbox(
    "🌐 Enable Live Web Search (supplement vector DB context)", 
    value=True,
    help="Fetches live web context alongside local ChromaDB context."
)

custom_instructions = st.text_area(
    "Custom Focus or Interlinking Instructions (Optional):",
    placeholder="e.g., Interlink this topic with judicial review, Article 102/191, landmark 10th schedule cases."
)

# Session State Initializations
if "max_estimate" not in st.session_state:
    st.session_state.max_estimate = None
if "reason" not in st.session_state:
    st.session_state.reason = ""
if "selected_count" not in st.session_state or st.session_state.selected_count < 1:
    st.session_state.selected_count = 10
if "analyzed_topic" not in st.session_state:
    st.session_state.analyzed_topic = None

# Capacity Estimation Logic
if st.button("🔍 Estimate Max Question Capacity"):
    with st.spinner("Analyzing topic depth with gpt-5.6-luna..."):
        db_blocks = []
        try:
            db_results = vector_db.similarity_search_with_score(topic, k=8)
            db_blocks = [
                f"[DB Source: {os.path.basename(doc.metadata.get('source', 'Doc'))}]\n{doc.page_content.strip()}"
                for doc, score in db_results if doc.page_content.strip()
            ]
        except Exception:
            pass
        
        web_blocks = ""
        if enable_web_search:
            web_blocks = fetch_web_context(topic)

        combined_context = "\n\n---\n\n".join(db_blocks)
        if web_blocks:
            combined_context += "\n\n--- LIVE WEB CONTEXT ---\n\n" + web_blocks

        st.session_state.last_context = combined_context

        estimate_prompt = f"""
You are an expert UPSC Civil Services examination paper setter.
Estimate the MAXIMUM total number of distinct, high-quality UPSC-level MCQs that can be created strictly on the topic "{topic}".

--- CONTEXT ---
{combined_context}
--------------

Return a single valid JSON object strictly in this structure:
{{"estimated_max": 50, "reason": "Detailed coverage of Tenth Schedule, 52nd Amendment, 91st Amendment, Speaker powers, and landmark judgments."}}
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
            
            st.session_state.max_estimate = max(10, int(data.get("estimated_max", 50)))
            st.session_state.reason = data.get("reason", "Analyzed topic scope successfully.")
            st.session_state.analyzed_topic = topic
            st.session_state.selected_count = st.session_state.max_estimate
        except Exception as e:
            st.error(f"❌ `{PRIMARY_MODEL}` API / Parse Error: {e}")

# Persistent UI Display Box
if st.session_state.max_estimate and st.session_state.analyzed_topic == topic:
    max_q = st.session_state.max_estimate
    st.success(f"🎯 **AI Capacity Estimation ({PRIMARY_MODEL})**: Up to **{max_q} distinct MCQs** can be generated.")
    st.caption(f"*Context Note: {st.session_state.reason}*")
    
    st.markdown("### Choose Question Volume:")
    opt_25 = max(5, round(max_q * 0.25))
    opt_50 = max(10, round(max_q * 0.50))
    opt_75 = max(15, round(max_q * 0.75))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button(f"📦 Quick: {opt_25}"):
            st.session_state.selected_count = opt_25
    with col2:
        if st.button(f"📘 Standard: {opt_50}"):
            st.session_state.selected_count = opt_50
    with col3:
        if st.button(f"📚 Large: {opt_75}"):
            st.session_state.selected_count = opt_75
    with col4:
        if st.button(f"🔥 Full: {max_q}"):
            st.session_state.selected_count = max_q

# Quantity Selector
safe_value = max(1, int(st.session_state.get("selected_count", 10)))
selected_count = st.number_input(
    "Selected Quantity to Generate:", 
    min_value=1, 
    max_value=1000, 
    value=safe_value,
    step=5
)
st.session_state.selected_count = selected_count

# Generation Pipeline
if st.button("🚀 Generate Question Bank", type="primary"):
    batch_size = 5
    generated_mcqs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    batches_count = (selected_count + batch_size - 1) // batch_size
    
    extra_instructions_prompt = ""
    if custom_instructions.strip():
        extra_instructions_prompt = f"\nSPECIAL FOCUS & INTERLINKING INSTRUCTIONS:\n{custom_instructions.strip()}\n"

    for b in range(batches_count):
        current_batch_qty = min(batch_size, selected_count - len(generated_mcqs))
        status_text.text(f"Generating batch {b+1} of {batches_count} ({current_batch_qty} questions) on '{topic}'...")
        
        # Retrieval with keyword/relevance filtering
        context = ""
        try:
            db_results = vector_db.similarity_search_with_score(topic, k=6)
            db_blocks = [
                f"[Source: {os.path.basename(doc.metadata.get('source', 'Doc'))}]\n{doc.page_content.strip()}"
                for doc, score in db_results if doc.page_content.strip()
            ]
            context = "\n\n---\n\n".join(db_blocks)
        except Exception:
            pass

        if enable_web_search or not context.strip():
            web_data = fetch_web_context(topic, max_results=3)
            if web_data.strip():
                context += "\n\n--- LIVE WEB CONTEXT ---\n\n" + web_data

        prompt = f"""
You are an expert UPSC Civil Services Examination paper setter.

PRIMARY MANDATE:
Generate {current_batch_qty} distinct, high-quality UPSC Prelims MCQs STRICTLY on the topic: "{topic}".
DO NOT generate questions about any other unrelated topic (e.g., UPSC body administration, Ramsar sites, Harappan culture, etc.). If the context contains off-topic material, IGNORE IT and rely on your core Indian Constitutional Polity knowledge on "{topic}".

--- REFERENCE CONTEXT ---
{context if context.strip() else "No local vector context available. Rely strictly on official UPSC syllabus knowledge for " + topic + "."}
------------------------
{extra_instructions_prompt}

FORMAT DIVERSITY (Rotate dynamically across these 18 UPSC Formats):
1. Single-Correct Statement
2. Standard Two-Statement (1 only, 2 only, Both 1 and 2, Neither 1 nor 2)
3. Standard Three-Statement Combination
4. Four-Statement Complex Combination
5. Pairs Matching / Match List-I with List-II
6. New UPSC Format: "How many of the above statements/pairs are correct?" (Only one, Only two, Only three, All four / None)
7. Assertion (A) and Reason (R)
8. Incorrect/False Statement Identification ("Which of the following is NOT correct?")
9. Landmark Supreme Court Case / Judgment Identification
10. Constitutional Article / Schedule / Amendment Identification
11. Situation-Based / Practical Application Scenario
12. Conceptual / Definition-Based Question
13. Chronological Sequence / Historical Evolution
14. Comparative Analysis (e.g., Union vs State, Pre-Amendment vs Post-Amendment)
15. Exception / Limitation / Safeguard Identification
16. Discretionary Powers / Role of Constitutional Functionaries
17. Interlinked / Multi-Domain Concept Integration
18. Passageway / Excerpt-Based Identification

GUIDELINES:
- Every question must be numbered sequentially starting from {len(generated_mcqs) + 1}.
- Provide options clear and unambiguous.
- Include a detailed "Answer" and comprehensive "Explanation" for each question.

Start output immediately with Question {len(generated_mcqs) + 1}:
"""
        try:
            res = client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": f"You are an expert UPSC Polity paper setter specializing in {topic}."},
                    {"role": "user", "content": prompt}
                ]
            )
            batch_output = res.choices[0].message.content
            generated_mcqs.append(batch_output)
            progress_bar.progress((b + 1) / batches_count)
        except Exception as e:
            st.error(f"❌ Stopping generation due to `{PRIMARY_MODEL}` API Error on batch {b+1}: {e}")
            break

    if generated_mcqs:
        status_text.success(f"Successfully generated question bank strictly using {PRIMARY_MODEL}!")
        full_test_paper = "\n\n---\n\n".join(generated_mcqs)
        st.markdown("## Generated Question Bank")
        st.markdown(full_test_paper)
        
        st.download_button(
            label="📥 Download Test Paper (.txt)",
            data=full_test_paper,
            file_name=f"UPSC_Polity_{topic.replace(' ', '_')}.txt",
            mime="text/plain"
        )
