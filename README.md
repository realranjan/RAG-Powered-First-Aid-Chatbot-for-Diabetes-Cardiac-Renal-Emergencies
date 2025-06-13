# RAG-Powered First-Aid Chatbot for Diabetes, Cardiac & Renal Emergencies

## Mission Statement
Build a patient-safety–aware chatbot that combines local knowledge embeddings with real-time web evidence to deliver actionable first-aid guidance, always prefaced with a clinical disclaimer.

**Disclaimer:** "This information is for educational purposes only and is not a substitute for professional medical advice."

## Core Objectives

1.  **Triage / Diagnosis:** Infer the most likely condition from free-text symptoms.
2.  **Hybrid Retrieval:**
    *   Local semantic search over the 60 provided medical snippets.
    *   Serper.dev web search for fresh evidence.
    *   Fuse results (semantic + Serper.dev results) and rank by relevance (keyword search).
3.  **Answer Generation:** Produce ≤ 250-word response with: condition, first-aid steps, key medicine(s), and source citations. All answers are always prefixed with a clinical disclaimer.

## Focus Areas

*   **Diabetes:** Type 1, Type 2, Gestational, ketoacidosis, hypoglycaemia.
*   **Cardiac:** Myocardial infarction, angina, arrhythmia, heart failure.
*   **Renal:** AKI, CKD, hyperkalaemia, dialysis crises.

## Project Structure

- `src/` — Main source code (including `medibot.py` for Streamlit UI, retrieval, and LLM logic)
- `tests/` — Automated tests (pytest, includes 10 sample queries)
- `data/` — Local data (e.g., Assignment Data Base.xlsx)
- `vectorstore/` — FAISS vector database
- `requirements.txt` — Python dependencies
- `README.md` — This file
- `architecture.md` — System architecture and design

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables**
   - Create a `.env` file in the root directory with:
     ```
     HF_TOKEN=your_huggingface_token
     SERPER_API_KEY=your_serper_dev_key
     ```
4. **Prepare the vectorstore**
   - Place your 60 medical snippets in `/data/Assignment Data Base.xlsx` (already present).
   - Run:
     ```bash
     python src/create_memory_for_llm.py
     ```
5. **Run the chatbot**
   ```bash
   streamlit run src/medibot.py
   ```
6. **Run automated tests**
   ```bash
   pytest tests/test_sample_queries.py
   ```

## Usage Instructions
- Open the Streamlit app in your browser (usually http://localhost:8501).
- Enter a query related to diabetes, cardiac, or renal emergencies (see sample queries below).
- The chatbot will:
  - Preface every answer with a clinical disclaimer (enforced in all code paths)
  - Infer the most likely condition
  - Provide first-aid steps and key medicines
  - Cite both local and web sources
  - Refuse to answer out-of-scope questions

**Sample Queries:**
- "I'm sweating, shaky, and my glucometer reads 55 mg/dL—what should I do right now?"
- "Crushing chest pain shooting down my left arm—do I chew aspirin first or call an ambulance?"
- (See `/tests/test_sample_queries.py` for all 10 test queries)

## Design Trade-offs
- **Retrieval:** Combines local semantic search (FAISS + MiniLM) and Serper.dev web search for up-to-date evidence.
- **Fusion & Ranking:** Uses keyword overlap for simple, transparent ranking. More advanced semantic ranking could be added.
- **LLM:** Uses Mistral-7B-Instruct-v0.3 via HuggingFace Endpoint for cost and performance balance.
- **Safety:** Strict domain restriction, always includes a clinical disclaimer, and refuses out-of-scope queries.
- **Testing:** Automated pytest script for the 10 sample queries ensures core requirements are met.

## Known Limitations
- No advanced semantic reranking (just keyword overlap).
- No user feedback loop or learning from corrections.
- Dependent on external APIs (HuggingFace, Serper.dev) for inference and web search.
- Performance and accuracy may vary with API/model changes. 