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

st.set_page_config(page_title="UPSC Polity MCQ Generator (V45 Agentic)", page_icon="📚", layout="wide")

# --- SAFE JSON CLEANING HELPER ---
def clean_json_string(raw_str: str) -> str:
    """Safely strip markdown code fences and whitespace from LLM JSON responses."""
    cleaned = raw_str.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()

# --- RESET SESSION FUNCTION ---
def reset_session():
    keys_to_clear = [
        "max_estimate", "reason", "selected_count", 
        "analyzed_topic", "last_context", "generated_paper", 
        "generated_questions_history", "audit_report_history"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- SAFE SECRETS RETRIEVAL ---
def get_secret(key_name: str) -> str:
    """Safely fetch secrets from Streamlit secrets or environment variables."""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.environ.get(key_name, "")

openai_key = get_secret("OPENAI_API_KEY")
anthropic_key = get_secret("ANTHROPIC_API_KEY")
gemini_key = get_secret("GEMINI_API_KEY")

openai_client = OpenAI(api_key=openai_key) if openai_key else None
claude_client = anthropic.Anthropic(api_key=anthropic_key) if anthropic_key else None
gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None

# --- SIDEBAR CONTROLS & MODEL SELECTOR ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    selected_provider = st.selectbox(
        "🧠 Select AI Model Engine:",
        [
            "OpenAI (gpt-5.6-luna)", 
            "Claude (claude-3-5-sonnet-20241022)", 
            "Claude (claude-3-opus-20240229)",
            "Gemini (gemini-2.5-pro)",
            "Gemini (gemini-2.5-flash)"
        ],
        index=0,
        help="Choose which LLM provider runs the 3-stage agentic pipeline."
    )
    
    st.info(f"Active Engine: **{selected_provider}**")
    st.markdown("---")
    
    st.header("🛠️ Session Controls")
    st.markdown("Clear session history and reset deduplication state for new topics.")
    if st.button("🔄 Reset / Clear Session", use_container_width=True):
        reset_session()

# MAIN HEADER
st.title("📚 UPSC Polity MCQ Generator (3-Stage Agentic Pipeline)")
st.caption(f"Powered by Constitution Bare Act, Indian Kanoon, Web Search & {selected_provider}")

def call_llm(provider: str, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    """Unified LLM router for OpenAI, Claude, and Gemini."""
    if "OpenAI" in provider:
        if not openai_client:
            raise ValueError("OPENAI_API_KEY is missing! Add it to environment variables or Streamlit secrets.")
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
            raise ValueError("ANTHROPIC_API_KEY is missing! Add it to environment variables or Streamlit secrets.")
        
        claude_model_id = "claude-3-5-sonnet-20241022" if "sonnet" in provider.lower() else "claude-3-opus-20240229"
        
        prompt_content = user_prompt
        if json_mode:
            prompt_content += "\n\nIMPORTANT: Output STRICT valid JSON only. Do not wrap in markdown or prose."
            
        res = claude_client.messages.create(
            model=claude_model_id,
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt_content}]
        )
        return res.content[0].text.strip()

    elif "Gemini" in provider:
        if not gemini_client:
            raise ValueError("GEMINI_API_KEY is missing! Add it to environment variables or Streamlit secrets.")
        
        gemini_model_id = "gemini-2.5-pro" if "pro" in provider.lower() else "gemini-2.5-flash"
        
        prompt = f"{system_prompt}\n\n{user_prompt}"
        if json_mode:
            prompt += "\n\nIMPORTANT: Output STRICT valid JSON only."
            
        res = gemini_client.models.generate_content(
            model=gemini_model_id,
            contents=prompt,
        )
        return res.text.strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

@st.cache_resource
def load_vector_db():
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        try:
            import ingest_docs
            ingest_docs.main()
        except Exception as e:
            st.warning(f"Auto-ingestion note: {e}")

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
if "audit_report_history" not in st.session_state:
    st.session_state.audit_report_history = []

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
{{"estimated_max": 50, "reason": "Detailed conceptual coverage including constitutional inferences, statutory mechanics, and landmark Supreme Court judgments across distinct UPSC question formats."}}
"""
        try:
            raw = call_llm(
                provider=selected_provider,
                system_prompt="You are an assistant that outputs strictly valid JSON.",
                user_prompt=estimate_prompt,
                json_mode=True
            )
            data = json.loads(clean_json_string(raw))
            
            st.session_state.max_estimate = max(5, int(data.get("estimated_max", 30)))
            st.session_state.reason = data.get("reason", "Analyzed topic scope successfully.")
            st.session_state.analyzed_topic = topic
            st.session_state.selected_count = st.session_state.max_estimate
        except Exception as e:
            st.error(f"❌ `{selected_provider}` API / Parse Error: {e}")

if st.session_state.max_estimate and st.session_state.analyzed_topic == topic:
    max_q = st.session_state.max_estimate
    st.success(f"🎯 **AI Capacity Estimation ({selected_provider})**: Up to **{max_q} distinct MCQs** can be generated.")
    st.caption(f"*Context Note: {st.session_state.reason}*")
    
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

safe_value = max(1, int(st.session_state.get("selected_count", 10)))
selected_count = st.number_input("Selected Quantity to Generate:", min_value=1, max_value=1000, value=safe_value, step=5)
st.session_state.selected_count = selected_count

# =========================================================
# AGENTIC PIPELINE STAGE FUNCTIONS
# =========================================================

def run_r1_retriever(micro_topic, requested_qty, custom_instr):
    """STAGE 1: R1 RETRIEVAL AGENT - Constructs Structured Evidence Package"""
    db_chunks = []
    try:
        db_results = vector_db.similarity_search_with_score(micro_topic, k=6)
        for doc, score in db_results:
            if doc.page_content.strip():
                db_chunks.append({
                    "source": os.path.basename(doc.metadata.get('source', 'Bare Act')),
                    "content": doc.page_content.strip()
                })
    except Exception:
        pass

    kanoon_data = fetch_kanoon_context(micro_topic, max_results=2) if enable_kanoon else ""
    web_data = fetch_web_context(micro_topic, max_results=2) if enable_web_search else ""

    r1_prompt = f"""
Act as a Senior Constitutional Researcher & Retrieval Planner.
Topic: "{micro_topic}"
Custom Focus: "{custom_instr}"

Raw Database Chunks: {json.dumps(db_chunks)}
Raw Kanoon Precedents: {kanoon_data}
Raw Live Web Data: {web_data}
Prior Generated Questions: {json.dumps(st.session_state.generated_questions_history)}

Extract and package all authoritative legal evidence into a single clean JSON structure:
{{
    "micro_topic": "{micro_topic}",
    "evidence_chunks": [
        {{
            "evidence_id": "EV-1",
            "source": "Source Name",
            "content": "Verbatim or detailed legal content..."
        }}
    ],
    "prior_questions_context": {json.dumps(st.session_state.generated_questions_history)}
}}
"""
    raw_res = call_llm(
        provider=selected_provider,
        system_prompt="You are a retrieval agent outputting valid JSON.",
        user_prompt=r1_prompt,
        json_mode=True
    )
    return json.loads(clean_json_string(raw_res))


def run_v45_generator(evidence_package, qty):
    """STAGE 2: V45 GENERATION AGENT - Generates Candidate Questions"""
    v45_system = "Act as a former UPSC Civil Services Examination Paper Setter and Senior Constitutional Law Professor."
    
    v45_prompt = f"""
UPSC CSE PRELIMS ELITE QUESTION GENERATOR (V45 ENGINE)

OBJECTIVE:
Generate exactly {qty} authentic UPSC CSE Prelims MCQs grounded STRICTLY in the provided evidence package.

EVIDENCE PACKAGE:
{json.dumps(evidence_package)}

MANDATORY RULES:
1. Ground every statement exclusively in evidence_chunks.
2. Check against prior_questions_context to avoid repeating already-tested concepts.
3. Tag every question with a competency: "identification" | "comparison" | "judicial_reasoning" | "institutional_relationship" | "evolution" | "application" | "exception".
4. Provide a full explanation (75–120 words) with elimination logic for every question.

Return strictly a JSON object with this key:
{{
    "candidate_questions": [
        {{
            "question_id": "Q-1",
            "topic": "{topic}",
            "competency": "judicial_reasoning",
            "question_text": "Consider the following statements...",
            "options": [
                {{"label": "a", "text": "1 only"}},
                {{"label": "b", "text": "2 only"}},
                {{"label": "c", "text": "Both 1 and 2"}},
                {{"label": "d", "text": "Neither 1 nor 2"}}
            ],
            "correct_option": "b",
            "explanation": "Statement 1 is incorrect because... Statement 2 is correct because...",
            "source_evidence_ids": ["EV-1"]
        }}
    ]
}}
"""
    raw_res = call_llm(
        provider=selected_provider,
        system_prompt=v45_system,
        user_prompt=v45_prompt,
        json_mode=True
    )
    return json.loads(clean_json_string(raw_res)).get("candidate_questions", [])


def run_r2_validator(candidate_questions, evidence_package):
    """STAGE 3: R2 VALIDATION AGENT - Independent Quality Control Audit"""
    r2_system = "Act as an Independent UPSC Quality Reviewer, Law Professor, and Legal Accuracy Auditor."
    
    r2_prompt = f"""
INDEPENDENT UPSC QUALITY & ACCURACY AUDIT (R2 VALIDATOR)

INSPECT Candidate Questions against Evidence Package & Session History:
Evidence Package: {json.dumps(evidence_package)}
Candidate Questions: {json.dumps(candidate_questions)}

Audit Criteria:
1. Constitutional / Statutory Accuracy against Evidence Package.
2. Exactly ONE indisputable correct answer.
3. No duplicate concepts tested in prior_questions_context or current batch.
4. Distractors represent realistic legal misconceptions (no trivial tricks).

Classify EACH question as APPROVED, REVISE, or REJECT.

Return strictly a JSON object:
{{
    "results": [
        {{
            "question_id": "Q-1",
            "decision": "APPROVED",
            "reasons": ["Factually precise against EV-1", "Unique angle tested"]
        }}
    ]
}}
"""
    raw_res = call_llm(
        provider=selected_provider,
        system_prompt=r2_system,
        user_prompt=r2_prompt,
        json_mode=True
    )
    return json.loads(clean_json_string(raw_res)).get("results", [])


# =========================================================
# MAIN GENERATION TRIGGER
# =========================================================

if st.button("🚀 Generate Question Bank", type="primary"):
    batch_size = 5
    st.session_state.generated_questions_history = []
    st.session_state.audit_report_history = []
    approved_questions_master = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    batches_count = (selected_count + batch_size - 1) // batch_size
    
    for b in range(batches_count):
        current_batch_qty = min(batch_size, selected_count - len(approved_questions_master))
        if current_batch_qty <= 0:
            break

        # Stage 1: Retrieval
        status_text.text(f"Batch {b+1}/{batches_count}: Running Stage 1 (R1 Retrieval Agent)...")
        try:
            evidence_pkg = run_r1_retriever(topic, current_batch_qty, custom_instructions)
        except Exception as e:
            st.error(f"❌ Stage 1 Retrieval Error: {e}")
            break

        # Stage 2: Generation
        status_text.text(f"Batch {b+1}/{batches_count}: Running Stage 2 (V45 Generation Engine)...")
        try:
            candidates = run_v45_generator(evidence_pkg, current_batch_qty)
        except Exception as e:
            st.error(f"❌ Stage 2 Generation Error: {e}")
            break

        # Stage 3: Independent Audit
        status_text.text(f"Batch {b+1}/{batches_count}: Running Stage 3 (R2 Quality & Duplicate Audit)...")
        try:
            audit_decisions = run_r2_validator(candidates, evidence_pkg)
            st.session_state.audit_report_history.extend(audit_decisions)
            
            # Filter only APPROVED questions
            approved_ids = {
                item["question_id"]: item 
                for item in audit_decisions 
                if item["decision"] == "APPROVED"
            }
            
            for q in candidates:
                if q["question_id"] in approved_ids:
                    approved_questions_master.append(q)
                    # Add to session deduplication history
                    st.session_state.generated_questions_history.append({
                        "question_id": q["question_id"],
                        "summary": q["question_text"][:120],
                        "competency": q.get("competency", "general")
                    })

        except Exception as e:
            st.error(f"❌ Stage 3 Validation Error: {e}")
            break

        progress_bar.progress((b + 1) / batches_count)

    status_text.success(f"Pipeline Execution Complete! {len(approved_questions_master)} Approved Questions Generated.")
    
    # Format Approved Questions into Final Output Paper
    formatted_paper_blocks = []
    for idx, q in enumerate(approved_questions_master, 1):
        opts = "\n".join([f"({opt['label']}) {opt['text']}" for opt in q['options']])
        block = f"### Question {idx}\n\n{q['question_text']}\n\n{opts}\n\n**Correct Option:** ({q['correct_option']})\n\n**Competency:** `{q.get('competency', 'N/A')}`\n\n**Explanation:**\n{q['explanation']}"
        formatted_paper_blocks.append(block)
        
    st.session_state.generated_paper = "\n\n---\n\n".join(formatted_paper_blocks)

# =========================================================
# DISPLAY RESULTS & AUDIT REPORT
# =========================================================

if "generated_paper" in st.session_state and st.session_state.generated_paper:
    tab1, tab2 = st.tabs(["📄 Approved Question Bank", "🔍 R2 Audit Report & Rejections"])
    
    with tab1:
        st.markdown("## Publication-Ready Question Bank")
        st.markdown(st.session_state.generated_paper)
        
        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            st.download_button(
                label="📥 Download Test Paper (.txt)",
                data=st.session_state.generated_paper,
                file_name=f"UPSC_Polity_{topic.replace(' ', '_')}_V45.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_d2:
            if st.button("🗑️ Clear Paper", use_container_width=True):
                del st.session_state.generated_paper
                st.session_state.generated_questions_history = []
                st.session_state.audit_report_history = []
                st.rerun()

    with tab2:
        st.markdown("## R2 Independent Quality Audit Log")
        st.caption("Inspect audit decisions, reasons, and rejected/flagged candidate questions.")
        if st.session_state.audit_report_history:
            st.json(st.session_state.audit_report_history)
        else:
            st.info("No audit logs available for current session.")
