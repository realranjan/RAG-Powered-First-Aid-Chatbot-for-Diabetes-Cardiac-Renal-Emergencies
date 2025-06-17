import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv, find_dotenv
import re
import traceback

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
    "In case of an emergency, always consult a qualified healthcare provider or call local emergency services immediately."
)

CUSTOM_PROMPT_TEMPLATE = f"""{CLINICAL_DISCLAIMER}

You are a First-Aid Chatbot trained to assist with the following medical emergencies only:
- **Diabetes:** Type 1, Type 2, Gestational Diabetes, Diabetic Ketoacidosis (DKA), Hypoglycaemia
- **Cardiac:** Myocardial Infarction (Heart Attack), Angina, Arrhythmia, Congestive Heart Failure
- **Renal:** Acute Kidney Injury (AKI), Chronic Kidney Disease (CKD), Hyperkalaemia, Dialysis-related Crises

**Instructions:**
From the user's free-text symptom description:
1. **Triage / Diagnosis:** Infer the most likely emergency condition (within the categories above) and start your answer with "Inferred Condition:".
2. **Answer Generation:** Provide a concise (≤ 250 words) response including:
   - Inferred Condition (as a bold heading)
   - First-Aid Steps (as a numbered list)
   - Key Medicine(s) (as a bullet list, only if relevant)
   - Sources (as a bulleted list of proper references/links, do not use Local[x] or Web[x])

**Important Rules:**
- Only output the final answer for the user — no greetings, internal reasoning, or extra commentary.
- Do not show internal citations like Local[x] or Web[x].
- Keep everything within 250 words.

Context: {{context}}
Question: {{question}}

Begin your response directly below.
"""

def truncate_to_word_limit(text, limit=240):
    words = text.split()
    if len(words) > limit:
        return ' '.join(words[:limit]) + '...'
    return text

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

def load_llm():
    huggingface_repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # User requested model
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

def get_answer_with_disclaimer(response):
    return f"{CLINICAL_DISCLAIMER}\n\n{response}"

def clean_citations(text):
    # Remove Local[x] and Web[x] citations
    text = re.sub(r"Local\[\d+\]:[^\n]*\n?", "", text)
    text = re.sub(r"Web\[\d+\]:[^\n]*\n?", "", text)
    return text

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
            result = truncate_to_word_limit(result, 250)
            result = clean_citations(result)
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()