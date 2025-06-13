import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv, find_dotenv
import re

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# --- Hybrid Retrieval ---
def get_local_snippets(retriever, query, k=3):
    docs = retriever.get_relevant_documents(query)[:k]
    return docs, [doc.page_content for doc in docs]

def get_web_snippets(query, k=3):
    serper_api_key = os.environ.get("SERPER_API_KEY")
    if not serper_api_key:
        raise ValueError("SERPER_API_KEY not set in environment variables or .env file.")
    serper = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
    results = serper.results(query)
    organic = results.get("organic", [])[:k]
    snippets = []
    for item in organic:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        snippets.append(f"{title}: {snippet} (Source: {link})")
    return snippets, organic

def fuse_and_rank(local_snippets, web_snippets, query=None):
    # Tokenize query for keyword matching
    if query is not None:
        query_keywords = set(re.findall(r'\w+', query.lower()))
    else:
        query_keywords = set()
    
    all_snippets = [(s, 'local') for s in local_snippets] + [(s, 'web') for s in web_snippets]
    
    # Score by keyword overlap
    def score(snippet):
        snippet_words = set(re.findall(r'\w+', snippet[0].lower()))
        return len(query_keywords & snippet_words)
    
    ranked = sorted(all_snippets, key=score, reverse=True)
    # Return only the snippet text, in ranked order
    return [s[0] for s in ranked]

def build_context(snippets):
    return "\n".join(snippets)

# --- Prompt Template ---
CLINICAL_DISCLAIMER = (
    "**Disclaimer:** This information is for educational purposes only and is not a substitute for professional medical advice. "
    "Always consult a qualified healthcare provider in an emergency."
)

CUSTOM_PROMPT_TEMPLATE = f"""{CLINICAL_DISCLAIMER}

You are a first-aid assistant for ONLY the following emergencies:
- Diabetes (Type 1, Type 2, Gestational, ketoacidosis, hypoglycaemia)
- Cardiac (Myocardial infarction, angina, arrhythmia, heart failure)
- Renal (AKI, CKD, hyperkalaemia, dialysis crises)

If the user's question is NOT about these, politely reply: "Sorry, I can only answer questions about diabetes, cardiac, or renal emergencies."

1. Triage/Diagnosis: Infer the most likely condition from the symptoms (if in scope).
2. First-aid: List immediate steps and key medicine(s) if any.
3. Citations: Cite sources for each fact.
If you don't know, say so. Do not make up answers.
Limit your answer to 250 words.

Context: {{context}}
Question: {{question}}

Start the answer directly. No small talk."""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

def load_llm():
    huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set in environment variables or .env file.")
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )
    return llm

# Add a function to check if the query is in scope
def is_in_scope(query):
    allowed_keywords = [
        "diabetes", "type 1", "type 2", "gestational", "ketoacidosis", "hypoglycaemia",
        "cardiac", "myocardial infarction", "angina", "arrhythmia", "heart failure",
        "renal", "aki", "ckd", "hyperkalaemia", "dialysis"
    ]
    query_lower = query.lower()
    return any(word in query_lower for word in allowed_keywords)

def get_answer_with_disclaimer(response):
    return f"{CLINICAL_DISCLAIMER}\n\n{response}"

# --- Streamlit UI ---
def main():
    st.title("First-Aid Chatbot: Diabetes, Cardiac & Renal Emergencies")
    st.markdown(CLINICAL_DISCLAIMER)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Describe your emergency or symptoms...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        # Out-of-scope check
        if not is_in_scope(prompt):
            out_of_scope_msg = (
                "Sorry, I can only answer questions about diabetes, cardiac, or renal emergencies "
                "(including: Type 1/2 diabetes, gestational diabetes, ketoacidosis, hypoglycaemia, "
                "myocardial infarction, angina, arrhythmia, heart failure, AKI, CKD, hyperkalaemia, dialysis crises)."
            )
            st.chat_message('assistant').markdown(out_of_scope_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': out_of_scope_msg})
            return
        try:
            vectorstore = get_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
            # Hybrid retrieval
            local_docs, local_snippets = get_local_snippets(retriever, prompt, k=3)
            web_snippets, web_docs = get_web_snippets(prompt, k=3)
            fused_snippets = fuse_and_rank(local_snippets, web_snippets, query=prompt)
            context = build_context(fused_snippets)
            # QA Chain (manual context)
            llm = load_llm()
            prompt_template = set_custom_prompt()
            prompt_input = prompt_template.format(context=context, question=prompt)
            response = llm.invoke(prompt_input)
            result = get_answer_with_disclaimer(response)
            # Build citations
            citations = []
            for i, doc in enumerate(local_docs):
                citations.append(f"Local[{i+1}]: {getattr(doc, 'metadata', {}).get('source', 'local snippet')}")
            for i, doc in enumerate(web_docs):
                citations.append(f"Web[{i+1}]: {doc.get('link', '')}")
            result_to_show = result + "\n\n**Citations:**\n" + "\n".join(citations)
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()