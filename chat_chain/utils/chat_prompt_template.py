QA_CHAIN_LLM_PROMPT_TEPLATE = """\
{prompt}

Chat History:
{chat_history}

Latest user input: {user_input}
"""

QA_CHAIN_LLM_PROMPT_WITH_RETRIEVER_TEMPLATE = QA_CHAIN_LLM_PROMPT_TEPLATE + """\n
Context retrieved based on user input:
{document_context}
"""
