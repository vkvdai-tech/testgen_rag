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
            cleaned_raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(cleaned_raw)
            
            st.session_state.max_estimate = max(5, int(data.get("estimated_max", 30)))
            st.session_state.reason = data.get("reason", "Analyzed topic scope successfully using Master Prompt V36 methodology.")
            st.session_state.analyzed_topic = topic
            st.session_state.selected_count = st.session_state.max_estimate
        except Exception as e:
            st.error(f"❌ `{selected_provider}` API / Parse Error: {e}")

if st.session_state.max_estimate and st.session_state.analyzed_topic == topic:
    max_q = st.session_state.max_estimate
    st.success(f"🎯 **AI Capacity Estimation ({selected_provider})**: Up to **{max_q} distinct MCQs** can be generated across 18 UPSC formats.")
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

if st.button("🚀 Generate Question Bank", type="primary"):
    batch_size = 5
    generated_mcqs = []
    st.session_state.generated_questions_history = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    batches_count = (selected_count + batch_size - 1) // batch_size
    
    extra_instructions_prompt = ""
    if custom_instructions.strip():
        extra_instructions_prompt = f"\nSPECIAL FOCUS & INTERLINKING INSTRUCTIONS:\n{custom_instructions.strip()}\n"

    SYSTEM_ROLE = """Act as a former UPSC Civil Services Examination Paper Setter, Constitutional Law Professor, and UPSC Psychometric Assessment Designer."""

    for b in range(batches_count):
        current_batch_qty = min(batch_size, selected_count - len(generated_mcqs))
        status_text.text(f"Generating batch {b+1} of {batches_count} ({current_batch_qty} questions) on '{topic}' using {selected_provider}...")
        
        context = ""
        try:
            db_results = vector_db.similarity_search_with_score(topic, k=6)
            db_blocks = [
                f"[Bare Act Source: {os.path.basename(doc.metadata.get('source', 'Doc'))}]\n{doc.page_content.strip()}"
                for doc, score in db_results if doc.page_content.strip()
            ]
            context = "\n\n---\n\n".join(db_blocks)
        except Exception:
            pass

        if enable_kanoon:
            kanoon_data = fetch_kanoon_context(topic, max_results=2)
            if kanoon_data.strip():
                context += "\n\n--- INDIAN KANOON PRECEDENTS ---\n\n" + kanoon_data

        if enable_web_search:
            web_data = fetch_web_context(topic, max_results=2)
            if web_data.strip():
                context += "\n\n--- LIVE WEB / PRS CONTEXT ---\n\n" + web_data

        history_str = ""
        if st.session_state.generated_questions_history:
            history_str = "\n\nCRITICAL DEDUPLICATION RULE:\nDo NOT reuse or restate any of the following statements or core premises already generated in previous questions:\n"
            for idx, q_summary in enumerate(st.session_state.generated_questions_history, 1):
                history_str += f"- Question Premise {idx}: {q_summary}\n"

        v36_prompt = f"""
UPSC CSE PRELIMS ELITE QUESTION BANK – MASTER PROMPT (V36)

OBJECTIVE:
Using the provided micro-topic "{topic}" as the complete syllabus, generate an authentic UPSC CSE Prelims question batch of {current_batch_qty} questions matching the reasoning, language, difficulty, and psychometric quality of UPSC Prelims (2018–2026).

STRICT TOPIC & NO-DUPLICATE MANDATE:
- Every question MUST be strictly focused on "{topic}".
- NO DUPLICATE STATEMENTS: Each individual statement across all questions MUST test a unique constitutional rule, exception, numerical threshold, or judicial precedent.
- DO NOT rephrase or repeat previously generated statements, facts, or scenarios.
{history_str}

--- CONSTITUTIONAL & JUDICIAL REFERENCE CONTEXT ---
{context if context.strip() else "Rely strictly on official Constitution of India Bare Act and Supreme Court precedents for " + topic + "."}
--------------------------------------------------
{extra_instructions_prompt}

MANDATORY 18-FORMAT SYSTEM REGISTRY:
You MUST dynamically rotate questions across the following 18 authentic UPSC CSE Prelims formats in this batch:
1. Classic 2-Statement Combination (1 only, 2 only, Both, Neither)
2. Classic 3-Statement Combination (1 and 2 only, 2 and 3 only, etc.)
3. 4-Statement High-Density Combination
4. New UPSC Pattern (Quantitative Statement Matching: "How many of the above statements are correct? Only one / Only two / Only three / All four")
5. Pair Matching (Traditional Column A vs Column B)
6. Pair Matching (New Pattern: "How many of the above pairs are correctly matched? Only one pair / Only two pairs / Only three pairs / All pairs")
7. Assertion-Reason (Standard: Both True & R is correct explanation, etc.)
8. Assertion-Reason (Constitutional Logic / Ratio)
9. Negative Statement Selection ("Which of the above statements is/are NOT correct?")
10. Constitutional Definition / Philosophical Concept (e.g. "What constitutes 'Liberty', 'Republic', or 'Due Process'?")
11. Application / Case-Scenario Based (Practical situation requiring legal interpretation)
12. Exception / Limitation Based (Identifying specific constitutional exceptions or non-applicability)
13. Chronological / Procedural Order (Sequencing constitutional steps, legislative passing, or emergency processes)
14. Constitutional Body vs Statutory Body / Non-Constitutional Distinction
15. Multi-Article Interlinking (Connecting Part III, Part IV, Part IVA, or Executive/Legislative powers)
16. Judicial Precedent & Ratio Matching (Linking SC judgments with constitutional principles)
17. Union vs. State Discretionary Power Distinction
18. Constitutional Amendment & Schedule Mapping

EXPLANATION REQUIREMENTS:
For every question, strictly provide:
1. Answer: Clearly stated option (e.g. Option (b)).
2. Explanation (75–120 words): Justify why the correct answer is correct, explain the key constitutional error in the incorrect options, and reinforce the constitutional principle and elimination logic.

Number questions starting from {len(generated_mcqs) + 1}.

Start output directly with Question {len(generated_mcqs) + 1}:
"""
        try:
            batch_output = call_llm(
                provider=selected_provider,
                system_prompt=SYSTEM_ROLE,
                user_prompt=v36_prompt
            )
            generated_mcqs.append(batch_output)
            
            extracted_summaries = re.findall(r"Question \d+:.*?(?=(Statement|Consider|With reference|Which|$))", batch_output, re.DOTALL)
            for summary in extracted_summaries:
                clean_sum = re.sub(r"\s+", " ", str(summary)).strip()[:120]
                if clean_sum:
                    st.session_state.generated_questions_history.append(clean_sum)

            progress_bar.progress((b + 1) / batches_count)
        except Exception as e:
            st.error(f"❌ Stopping generation due to `{selected_provider}` API Error on batch {b+1}: {e}")
            break

    if generated_mcqs:
        status_text.success(f"Successfully generated question bank using {selected_provider}!")
        full_test_paper = "\n\n---\n\n".join(generated_mcqs)
        st.session_state.generated_paper = full_test_paper

if "generated_paper" in st.session_state:
    st.markdown("## Generated Question Bank")
    st.markdown(st.session_state.generated_paper)
    
    col_d1, col_d2 = st.columns([3, 1])
    with col_d1:
        st.download_button(
            label="📥 Download Test Paper (.txt)",
            data=st.session_state.generated_paper,
            file_name=f"UPSC_Polity_{topic.replace(' ', '_')}_V36.txt",
            mime="text/plain",
            use_container_width=True
        )
    with col_d2:
        if st.button("🗑️ Clear Paper", use_container_width=True):
            del st.session_state.generated_paper
            if "generated_questions_history" in st.session_state:
                st.session_state.generated_questions_history = []
            st.rerun()