import os
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

topic = st.text_input("Enter Topic or Question Area:", "Basic Structure doctrine and Kesavananda Bharati case")

if st.button("Generate MCQ", type="primary"):
    with st.spinner("Retrieving context from polity corpus..."):
        results = vector_db.similarity_search_with_score(topic, k=4)
        context_blocks = []
        for doc, score in results:
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "N/A")
            context_blocks.append(f"[Source: {source}, Page: {page}]\n{doc.page_content.strip()}")
        context = "\n\n---\n\n".join(context_blocks)

    with st.spinner("Generating question with OpenAI..."):
        prompt = f"""
You are an expert UPSC Civil Services examination paper setter specializing in Indian Polity.
Based strictly on the provided reference context below, create 1 high-quality UPSC-style Multiple Choice Question (MCQ).

--- REFERENCE CONTEXT ---
{context}
------------------------

Guidelines for Question Creation:
1. Use standard UPSC formatting (2 or 3 statements, followed by "Which of the statements given above is/are correct?").
2. Options pattern: (a) 1 only, (b) 2 only, (c) Both 1 and 2, (d) Neither 1 nor 2.
3. Include detailed explanation covering each statement.

Format:
**Question:**
...
**Options:**
...
**Correct Answer:** ...
**Explanation:**
...
"""
     try:
    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
            {"role": "system", "content": "You are a precise UPSC examination paper setter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
except Exception as e:
    st.warning("`gpt-5.6-luna` endpoint rejected. Falling back to standard model...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise UPSC examination paper setter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

        st.markdown(response.choices[0].message.content)

    with st.expander("🔍 View Retrieved Context Chunks"):
        st.text(context)
