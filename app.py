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

st.set_page_config(page_title="UPSC Polity MCQ Generator (V36 Elite)", page_icon="📚", layout="centered")

st.title("📚 UPSC Polity MCQ Generator")
st.caption("Powered by ChromaDB, Web Search & OpenAI (gpt-5.6-luna) | Master Prompt V36 Engine")

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
        st.warning(f"⚠️ Live web retrieval paused ({e}). Continuing with LLM depth evaluation.")
        return ""

# Inputs
topic = st.text_input("Enter Micro-Topic / Primary Topic:", "Anti-Defection Law")

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
You are an expert UPSC Civil Services examination paper setter, Constitutional Law Professor, and UPSC Psychometric Assessment Designer.
Estimate the MAXIMUM total number of distinct, high-quality, non-repetitive UPSC-level MCQs that can be created STRICTLY on the micro-topic "{topic}".

--- CONTEXT ---
{combined_context}
--------------

Return a single valid JSON object strictly in this structure:
{{"estimated_max": 50, "reason": "Detailed conceptual coverage including constitutional inferences, statutory mechanics, and landmark Supreme Court judgments."}}
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
            
            st.session_state.max_estimate = max(5, int(data.get("estimated_max", 30)))
            st.session_state.reason = data.get("reason", "Analyzed topic scope successfully using Master Prompt V36 methodology.")
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
    opt_25 = max(3, round(max_q * 0.25))
    opt_50 = max(5, round(max_q * 0.50))
    opt_75 = max(8, round(max_q * 0.75))
    
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

# Generation Pipeline using Master Prompt V36
if st.button("🚀 Generate Question Bank", type="primary"):
    batch_size = 5
    generated_mcqs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    batches_count = (selected_count + batch_size - 1) // batch_size
    
    extra_instructions_prompt = ""
    if custom_instructions.strip():
        extra_instructions_prompt = f"\nSPECIAL FOCUS & INTERLINKING INSTRUCTIONS:\n{custom_instructions.strip()}\n"

    SYSTEM_ROLE = """Act as a former UPSC Civil Services Examination Paper Setter, Constitutional Law Professor, and UPSC Psychometric Assessment Designer."""

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

        v36_prompt = f"""
UPSC CSE PRELIMS ELITE QUESTION BANK – MASTER PROMPT (V36)

OBJECTIVE:
Using the provided micro-topic "{topic}" as the complete syllabus, generate an authentic UPSC CSE Prelims question batch of {current_batch_qty} questions matching the reasoning, language, difficulty, and psychometric quality of UPSC Prelims (2018–2026).

STRICT TOPIC MANDATE:
- Every question MUST be strictly focused on "{topic}".
- DO NOT generate questions about any unrelated topic (e.g. UPSC organization administrative functions, NDA/CDS notifications, Ramsar sites, Harappan civilization, etc.).
- If the provided reference context contains irrelevant information, DISREGARD THE CONTEXT and rely strictly on your expert knowledge of the Constitution of India and UPSC CSE syllabus regarding "{topic}".

--- REFERENCE CONTEXT ---
{context if context.strip() else "No local reference text found. Rely strictly on official Constitution of India and UPSC syllabus knowledge for " + topic + "."}
------------------------
{extra_instructions_prompt}

KNOWLEDGE BASE & SOURCE HIERARCHY:
1. Constitution of India (Bare Act, as amended)
2. Supreme Court Judgments
3. Official Government Sources (Constituent Assembly Debates, Parliamentary Documents, Law Commission Reports, PRS)
4. Standard UPSC References (M. Laxmikanth, D.D. Basu, Subhash Kashyap, NCERT)
5. Previous UPSC Prelims Papers (2011–2026) for language, reasoning, and elimination techniques.

QUESTION DESIGN & FORMATS:
Design each question around a unique constitutional inference (purpose, scope, limitation, exception, evolution, judicial interpretation).
Rotate dynamically across authentic UPSC formats:
- Statement-Based Questions (2-6 statements) using options like 1 only, 2 only, Both 1 and 2, Neither 1 nor 2, 1 and 2 only, 1, 2 and 3, etc.
- Pair-Based Questions / Matching using options like Only one pair, Only two pairs, Only three pairs, All pairs, No pair, etc.
- New UPSC Format: "How many of the above statements are correct?" (Only one, Only two, Only three, All four)
- Assertion-Reason
- Constitutional Scenario / Application-Based
- Landmark SC Judgment / Constitutional Principles

EXPLANATION REQUIREMENTS:
For every question, strictly provide:
1. Answer: Clearly stated option.
2. Explanation (75–120 words): Justify why the correct answer is correct, explain the key constitutional error in the incorrect options, and reinforce the constitutional principle and elimination logic.

Number questions starting from {len(generated_mcqs) + 1}.

Start output directly with Question {len(generated_mcqs) + 1}:
"""
        try:
            res = client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_ROLE},
                    {"role": "user", "content": v36_prompt}
                ]
            )
            batch_output = res.choices[0].message.content
            generated_mcqs.append(batch_output)
            progress_bar.progress((b + 1) / batches_count)
        except Exception as e:
            st.error(f"❌ Stopping generation due to `{PRIMARY_MODEL}` API Error on batch {b+1}: {e}")
            break

    if generated_mcqs:
        status_text.success(f"Successfully generated question bank strictly using {PRIMARY_MODEL} (Master Prompt V36)!")
        full_test_paper = "\n\n---\n\n".join(generated_mcqs)
        st.markdown("## Generated Question Bank")
        st.markdown(full_test_paper)
        
        st.download_button(
            label="📥 Download Test Paper (.txt)",
            data=full_test_paper,
            file_name=f"UPSC_Polity_{topic.replace(' ', '_')}_V36.txt",
            mime="text/plain"
        )
