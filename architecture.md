# Architecture of RAG-Powered First-Aid Chatbot

## Overview
This document outlines the architecture of the RAG-Powered First-Aid Chatbot, designed to provide first-aid guidance for diabetes, cardiac, and renal emergencies.

## Components

### 1. Data Ingestion & Vectorization
*   **Purpose:** To process medical knowledge snippets and convert them into a searchable format.
*   **Input:** Local corpus (60 pre-approved medical sentences) and potentially other structured medical data.
*   **Process:** Documents are loaded, chunked (if necessary), and then embedded using a HuggingFace embedding model. These embeddings are stored in a FAISS vector store.
*   **Output:** `vectorstore/db_faiss` containing the vectorized knowledge base.

### 2. Hybrid Retrieval System
*   **Purpose:** To retrieve relevant information from both local knowledge and real-time web sources.
*   **Components:**
    *   **Local Semantic Search:** Queries the FAISS vector store to find semantically similar medical snippets.
    *   **Web Search (Serper.dev):** Integrates with Serper.dev API to perform real-time web searches for fresh evidence.
    *   **Fusion & Keyword-Based Ranking:** Combines results from both local and web searches, then ranks them by keyword overlap with the user's query to provide the most relevant information.
*   **Input:** User query.
*   **Output:** Ranked list of relevant text snippets and their sources.

### 3. Language Model (LLM) for Answer Generation
*   **Purpose:** To generate concise, actionable first-aid guidance based on retrieved information and user queries.
*   **Model:** HuggingFaceEndpoint (e.g., Mistral-7B-Instruct-v0.3).
*   **Process:** The LLM receives the user's question, the ranked context (from hybrid retrieval), and a system prompt/template. It performs triage/diagnosis, synthesizes the answer, and ensures adherence to length and content requirements. **All answers are always prepended with the clinical disclaimer, enforced in all code paths.**
*   **Output:** Generated first-aid response, including condition, steps, medicines, and citations.

### 4. User Interface (Streamlit)
*   **Purpose:** Provides an interactive web interface for users to interact with the chatbot.
*   **Features:**
    *   Chat input for user queries.
    *   Display of chatbot responses.
    *   Incorporation of clinical disclaimer (always present).
    *   Chat history management.
    *   Strict domain restriction and out-of-scope handling.
*   **Framework:** Streamlit.

### 5. Automated Testing
*   **Purpose:** Ensure the chatbot meets project requirements and passes the 10 sample queries.
*   **Implementation:** Pytest script in `/tests/` runs all 10 queries and checks for disclaimer, length, and answer presence.

## Data Flow
1.  **User Query:** Input through the Streamlit UI.
2.  **Domain Check:** Query is checked for allowed domains (diabetes, cardiac, renal). Out-of-scope queries are politely refused.
3.  **Hybrid Retrieval:** The query is used to perform both local semantic search and Serper.dev web search.
4.  **Result Fusion & Ranking:** Retrieved documents from both sources are combined and ranked by keyword overlap with the query.
5.  **LLM Input:** The top-ranked documents, along with the user query and a predefined prompt, are fed to the LLM.
6.  **Answer Generation:** The LLM processes the input and generates a first-aid response.
7.  **Disclaimer Prefix:** The clinical disclaimer is always prepended to the generated answer (enforced in all code paths).
8.  **Display:** The final answer (with disclaimer and citations) is displayed in the Streamlit UI.

## Security & Safety Considerations
*   **Clinical Disclaimer:** Crucial for patient safety, prominently displayed and always included in every answer.
*   **Strict Domain Restriction:** Only answers questions about diabetes, cardiac, or renal emergencies. Out-of-scope queries are refused.
*   **API Key Management:** Hugging Face and Serper.dev API keys are managed securely using environment variables (`.env` file).
*   **Content Filtering:** The LLM is prompted to stick to the provided context and not hallucinate, and to prioritize patient safety.

## Automated Testing
*   **Pytest Coverage:** Automated test script runs all 10 sample queries and checks for disclaimer and length.

## Limitations & Future Improvements
*   **Limitations:**
    *   Only keyword-based ranking (no advanced semantic reranking).
    *   Strictly limited to diabetes, cardiac, and renal emergencies.
    *   Dependent on external APIs (HuggingFace, Serper.dev).
*   **Possible Future Improvements:**
    *   Add semantic reranking for even better relevance.
    *   Expand to more medical domains.
    *   Add user feedback and learning loop.
    *   More robust performance and latency monitoring.

## Future Enhancements (Beyond initial scope)
*   More sophisticated triage logic.
*   Integration with medical APIs for real-time drug interactions or dosage information.
*   User feedback mechanism for continuous improvement.
*   Deployment to cloud platforms. 