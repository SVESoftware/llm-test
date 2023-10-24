'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
from langchain.llms import CTransformers
from dotenv import find_dotenv, load_dotenv
import box
import os
import yaml
from ctransformers import AutoModelForCausalLM

import torch

print("CUDA: ", torch.cuda.is_available())

# Load environment variables from .env file
load_dotenv(find_dotenv())

CONFIG_FILE = os.environ.get('CONFIG_FILE', 'config/config.yml')
with open(CONFIG_FILE, 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def build_llm():
    # Local CTransformers model
    llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                model_type=cfg.MODEL_TYPE,
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
                model_type=cfg.MODEL_TYPE,
                config={'max_new_tokens': cfg.MAX_NEW_TOKENS,
                        'temperature': cfg.TEMPERATURE,
                        'threads': cfg.THREADS,
                        'context_length': 4096}
                )
    return llm

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def build_gpu_llm():
        n_gpu_layers = 6  # Metal set to 1 is enough.
        n_batch = 256  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
                        model_path=cfg.MODEL_BIN_PATH,
                        n_gpu_layers=n_gpu_layers,
                        n_batch=n_batch,
                        n_ctx=2048,
                        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
                        callback_manager=callback_manager,
                        verbose=True,
                        temperature=cfg.TEMPERATURE,
                        model_kwargs={
                                'max_new_tokens': cfg.MAX_NEW_TOKENS,
                        }
                )