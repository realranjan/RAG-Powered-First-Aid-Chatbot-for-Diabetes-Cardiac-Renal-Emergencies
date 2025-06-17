import os
import re

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mixtral-8x7B-Instruct-v0.1"  # User requested model

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
        
        
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain
CLINICAL_DISCLAIMER = (
    "**Disclaimer:** This information is for educational purposes only and is not a substitute for professional medical advice. "
    "In case of an emergency, always consult a qualified healthcare provider or call local emergency services immediately."
)

CUSTOM_PROMPT_TEMPLATE = """
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

Context: {context}
Question: {question}

Begin your response directly below.
"""

def clean_citations(text):
    # Remove Local[x] and Web[x] citations
    text = re.sub(r"Local\[\d+\]:[^\n]*\n?", "", text)
    text = re.sub(r"Web\[\d+\]:[^\n]*\n?", "", text)
    return text

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

def truncate_to_word_limit(text, limit=240):
    words = text.split()
    if len(words) > limit:
        return ' '.join(words[:limit]) + '...'
    return text

def get_answer_with_disclaimer(response):
    answer = "**Disclaimer:** This information is for educational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider in an emergency.\n\n" + response
    return truncate_to_word_limit(clean_citations(answer), 250)

if __name__ == "__main__":
    # Now invoke with a single query
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({'query': user_query})
    print("RESULT: ", get_answer_with_disclaimer(response["result"]))
    print("SOURCE DOCUMENTS: ", response["source_documents"])
