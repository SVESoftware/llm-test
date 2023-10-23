'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml

from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationChain

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import qa_template, custom_template, QA_PROMPT_HISTORY, QA_PROMPT_NEW, QA_PROMPT_HISTORY_NEW, QA_PROMPT_MISTRAL
from src.llm import build_llm, build_c_llm

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt

def set_conversational_qa_prompt():
    """
    """
    prompt = QA_PROMPT_MISTRAL
    return prompt


def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                       return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return dbqa

def build_conversational_retrieval_qa(llm, prompt, vectordb):


    """dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       #retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                       #return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                       chain_type_kwargs={'prompt': prompt, "memory": memory, 'verbose':cfg.VERBOSE}
                                       )
    """
    memory = ConversationBufferMemory()
    dbqa = ConversationChain(
    llm=llm, verbose=cfg.VERBOSE, prompt=prompt, memory=memory)


    return dbqa, memory


def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = build_llm()
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa

def setup_conversational_dbqa():
    #encode_kwargs = {'normalize_embeddings': True}
    # encode_kwargs = {}

    embeddings = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)

    llm = build_c_llm()
    print('llm', llm)
    qa_prompt = set_conversational_qa_prompt()
    dbqa, memory = build_conversational_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa, memory
