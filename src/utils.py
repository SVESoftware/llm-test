'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml
import os

from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationChain

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import qa_template
from src.prompts import PROMPTS
from src.llm import build_llm, build_c_llm, build_gpu_llm

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ConversationBufferWindowMemory

CONFIG_FILE = os.environ.get('CONFIG_FILE', 'config/config.yml')
with open(CONFIG_FILE, 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_prompt():
    """
    """
    return PROMPTS.get(cfg.PROMPT)


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
    memory = ConversationBufferWindowMemory()
    dbqa = ConversationChain(
    llm=llm, verbose=cfg.VERBOSE, prompt=prompt, memory=memory)

    """
    chain = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0),
    vectorstore.as_retriever(), # see below for vectorstore definition
    memory=memory,
    condense_question_prompt=condense_prompt,
    combine_docs_chain_kwargs=dict(prompt=combine_docs_custom_prompt)
)
    """


    return dbqa, memory

def build_stuff_retrieval_qa(llm, prompt, vectordb):


    memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="question", return_messages=True)

    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                       return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                       chain_type_kwargs={'prompt': prompt, 'verbose':cfg.VERBOSE, 'memory':memory}
                                       )

    return dbqa, memory


def setup_conversational_dbqa():


    if cfg.INTERNET:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cpu'})
    else:
        embeddings = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cpu'})


    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)

    if cfg.GPU:
        llm = build_gpu_llm()
    else:
        llm = build_c_llm()
    print('llm', llm)
    qa_prompt = set_prompt()
    dbqa, memory = build_conversational_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa, memory

def setup_stuff_dbqa():


    if cfg.INTERNET:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cuda'})
    else:
        embeddings = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cpu'})


    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)

    if cfg.GPU:
        llm = build_gpu_llm()
    else:
        llm = build_c_llm()
    print('llm', llm)
    qa_prompt = set_prompt()
    dbqa, memory = build_stuff_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa, memory