'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
from langchain.llms import CTransformers
from dotenv import find_dotenv, load_dotenv
import box
import yaml
from ctransformers import AutoModelForCausalLM

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def build_llm():
    # Local CTransformers model
    llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                model_type="mistral",
                config={'max_new_tokens': cfg.MAX_NEW_TOKENS,
                        'temperature': cfg.TEMPERATURE,
                        'threads': cfg.THREADS,
                        'context_length': 100}
                )


    #llm = AutoModelForCausalLM.from_pretrained(cfg.MODEL_BIN_PATH, model_type="mistral")

    return llm

def build_c_llm():
    # Local CTransformers model
    llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                model_type="mistral",
                config={'max_new_tokens': cfg.MAX_NEW_TOKENS,
                        'temperature': cfg.TEMPERATURE,
                        'threads': cfg.THREADS,
                        'context_length': 4096}
                )
    return llm
