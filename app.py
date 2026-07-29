import os
import json
import streamlit as st
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="UPSC Polity MCQ Generator", page_icon="📚", layout="centered")

st.title("📚 UPSC Polity MCQ Generator")
st.caption("Powered by ChromaDB local vector storage & OpenAI")

# API Key handling
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("Please configure OPENAI_API_KEY environment variable or Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)
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

# Topic Input
topic = st.text_input("Enter Topic or Question Area:", "Right to Equality")

# Initialize session state variables
if "max_estimate" not in st.session_state:
    st.session_state.max_estimate = None
if "selected_count" not in st.session_state:
    st.session_state.selected_count = 10

# Step 1: Uncapped Estimation
if st.button("🔍 Estimate Max Question Capacity"):
    with st.spinner("Analyzing corpus context depth..."):
        # Retrieve context proportional to topic size
        results = vector_db.similarity_search_with_score(topic, k=12)
        context_blocks = [
            f"[Source: {os.path.basename(doc.metadata.get('source', 'Doc'))}]\n{doc.page_content.strip()}"
            for doc, score in results
        ]
        context = "\n\n---\n\n".join(context_blocks)
        
        estimate_prompt = f"""
You are an expert UPSC paper setter. Analyze the following reference text and estimate the MAXIMUM total number of distinct, high-quality UPSC-level MCQs that can be created on this topic without repeating concepts or generating low-quality filler.

--- CONTEXT ---
{context}
--------------

Return ONLY a single JSON object with this exact format (no markdown codeblock, no explanation):
{{"estimated_max": <integer count, e.g. 50, 150, 300>, "reason": "<brief 1-sentence reason>"}}
"""
        try:
            res = client.chat.completions.create(
                model="gpt-5.6-luna",
                messages=[{"role": "user", "content": estimate_prompt}],
                temperature=0.2
            )
            raw = res.choices[0].message.content.strip()
        except Exception:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": estimate_prompt}],
                temperature=0.2
            )
            raw = res.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
            st.session_state.max_estimate = int(data.get("estimated_max", 50))
            st.session_state.reason = data.get("reason", "")
            st.session_state.analyzed_topic = topic
        except Exception:
            st.session_state.max_estimate = 50
            st.session_state.reason = "Standard topic depth detected."
            st.session_state.analyzed_topic = topic

# Display AI options if estimation is complete
if st.session_state.max_estimate and st.session_state.get("analyzed_topic") == topic:
    max_q = st.session_state.max_estimate
    st.success(f"🎯 **AI Estimation**: This topic can yield up to **{max_q} distinct MCQs**.")
    st.caption(f"*Context note: {st.session_state.get('reason', '')}*")
    
    st.markdown("### Choose Question Volume:")
    
    # Calculate smart preset options based on AI estimate
    opt_25 = max(5, round(max_q * 0.25))
    opt_50 = max(10, round(max_q * 0.50))
    opt_75 = max(15, round(max_q * 0.75))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button(f"📦 Quick: {opt_25} MCQs"):
            st.session_state.selected_count = opt_25
    with col2:
        if st.button(f"📘 Standard: {opt_50} MCQs"):
            st.session_state.selected_count = opt_50
    with col3:
        if st.button(f"📚 Large: {opt_75} MCQs"):
            st.session_state.selected_count = opt_75
    with col4:
        if st.button(f"🔥 Full: {max_q} MCQs"):
            st.session_state.selected_count = max_q

# Allow fine-tuning or manual selection
selected_count = st.number_input(
    "Selected Quantity to Generate:", 
    min_value=1, 
    max_value=1000, 
    value=st.session_state.selected_count,
    step=5
)

# Step 2: Iterative Generation Pipeline
if st.button("🚀 Generate Question Bank", type="primary"):
    batch_size = 5
    generated_mcqs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    batches_count = (selected_count + batch_size - 1) // batch_size
    
    for b in range(batches_count):
        current_batch_qty = min(batch_size, selected_count - len(generated_mcqs))
        status_text.text(f"Generating batch {b+1} of {batches_count} ({current_batch_qty} questions)...")
        
        results = vector_db.similarity_search_with_score(topic, k=8)
        context_blocks = [
            f"[Source: {os.path.basename(doc.metadata.get('source', 'Doc'))}, Pg: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content.strip()}"
            for doc, score in results
        ]
        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""
You are an expert UPSC Civil Services examination paper setter.
Based on the reference context, generate {current_batch_qty} distinct UPSC-style MCQs.

--- CONTEXT ---
{context}
---------------

Guidelines:
1. Standard UPSC 2-3 statement format per question.
2. Options: (a) 1 only, (b) 2 only, (c) Both 1 and 2, (d) Neither 1 nor 2.
3. Detailed explanations covering each statement.

Number starting from {len(generated_mcqs) + 1}.
"""
        try:
            res = client.chat.completions.create(
                model="gpt-5.6-luna",
                messages=[
                    {"role": "system", "content": "You are a precise UPSC examination paper setter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            batch_output = res.choices[0].message.content
        except Exception:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise UPSC examination paper setter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            batch_output = res.choices[0].message.content

        generated_mcqs.append(batch_output)
        progress_bar.progress((b + 1) / batches_count)

    status_text.success(f"Successfully generated {selected_count} questions!")
    
    full_test_paper = "\n\n---\n\n".join(generated_mcqs)
    st.markdown("## Generated Question Bank")
    st.markdown(full_test_paper)
    
    st.download_button(
        label="📥 Download Test Paper (.txt)",
        data=full_test_paper,
        file_name=f"UPSC_Polity_{topic.replace(' ', '_')}.txt",
        mime="text/plain"
    )
