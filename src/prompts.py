'''
===========================================
        Module: Prompts collection
===========================================
'''
# Note: Precise formatting of spacing and indentation of the prompt template is important for Llama-2-7B-Chat,
# as it is highly sensitive to whitespace changes. For example, it could have problems generating
# a summary from the pieces of context if the spacing is not done correctly
from langchain.prompts.prompt import PromptTemplate

qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in German language.' If you do not know the answer reply with 'I am sorry'.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on
the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""

system_prompt_2 = """Given the following context and chat history, answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer."""


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
SYSTEM_PROMPT_2 = B_SYS + system_prompt_2 + E_SYS

instruction = """
Context: {history} \n {context}
User question: {question}
Only return the helpful answer below on original language and nothing else.
Helpful answer:"""

prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
QA_PROMPT_HISTORY = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)


instruction = """
Context: {context}
User question: {question}
Only return the helpful answer below on original language and nothing else.
Helpful answer:"""

prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
QA_PROMPT_NEW = PromptTemplate(input_variables=["context", "question"], template=prompt_template)



instruction = """
Chat history: {history}
Context: {context}
User question: {question}
Only return the helpful answer below in Croatian language and nothing else.
Helpful answer:"""

prompt_template = B_INST + SYSTEM_PROMPT_2 + instruction + E_INST
QA_PROMPT_HISTORY_NEW = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)

B = "<s>"

instruction = """{history}
{input}
"""

prompt_template = B + B_INST + instruction + E_INST

QA_PROMPT_MISTRAL = PromptTemplate(input_variables=["input", "history"], template=prompt_template)
