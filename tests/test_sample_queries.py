import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from medibot import load_llm, set_custom_prompt, fuse_and_rank, build_context, get_vectorstore, get_local_snippets, get_web_snippets, get_answer_with_disclaimer

sample_queries = [
    "I'm sweating, shaky, and my glucometer reads 55 mg/dL—what should I do right now?",
    "My diabetic father just became unconscious; we think his sugar crashed. What immediate first-aid should we give?",
    "A pregnant woman with gestational diabetes keeps getting fasting readings around 130 mg/dL. What does this mean and how should we manage it?",
    "Crushing chest pain shooting down my left arm—do I chew aspirin first or call an ambulance?",
    "I'm having angina; how many nitroglycerin tablets can I safely take and when must I stop?",
    "Grandma has chronic heart failure, is suddenly short of breath, and her ankles are swelling. Any first-aid steps before we reach the ER?",
    "After working in the sun all day I've barely urinated and my creatinine just rose 0.4 mg/dL—could this be acute kidney injury and what should I do?",
    "CKD patient with a potassium level of 6.1 mmol/L—what emergency measures can we start right away?",
    "I took ibuprofen for back pain; now my flanks hurt and I'm worried about kidney damage—any immediate precautions?",
    "Type 2 diabetic, extremely thirsty, glucose meter says 'HI' but urine ketone strip is negative—what's happening and what's the first-aid?"
]

@pytest.mark.parametrize("query", sample_queries)
def test_sample_query(query):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    local_docs, local_snippets = get_local_snippets(retriever, query, k=3)
    web_snippets, web_docs = get_web_snippets(query, k=3)
    fused_snippets = fuse_and_rank(local_snippets, web_snippets, query=query)
    context = build_context(fused_snippets)
    llm = load_llm()
    prompt_template = set_custom_prompt()
    prompt_input = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt_input)
    response = get_answer_with_disclaimer(response)
    # Basic checks
    assert "Disclaimer" in response or "disclaimer" in response.lower()
    assert len(response.split()) <= 270  # Allowing a little buffer for word count 