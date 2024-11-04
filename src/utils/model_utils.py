import dspy
import torch
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.models.models_dict import MODELS_DICT
from src.config import Config
from src.models.llm_wrapper import LLM_WRAPPER

from llama_index.llms.groq import Groq

config = Config()

GENERATION_CONFIG = config.generation_config
LLAMACPP_GENERATION_CONFIG = config.llamacpp_generation_config
MODEL_CONFIG = config.model_config
LLAMACPP_MODEL_CONFIG = config.llamacpp_model_config
TOKENIZER_CONFIG = config.tokenizer_config

groq_id_to_hf_id = {"llama3-8b-8192": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                   "llama3-70b-8192": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                   "llama3-groq-70b-8192-tool-use-preview": "Groq/Llama-3-Groq-70B-Tool-Use",
                   "llama3-groq-8b-8192-tool-use-preview": "Groq/Llama-3-Groq-8B-Tool-Use"
                   }
def initialize_groq_model(model_name):
    groq_llm = Groq(model=model_name, tokenizer=groq_id_to_hf_id[model_name])
    return groq_llm
    
def load_hf_model(llm_name=config.hf_tiny_model, embed_model_name=config.hf_embed_model, embed_model_only=True):
    
    use_cuda = torch.cuda.is_available()

    model_name = MODELS_DICT[llm_name].get("awq",
                                    MODELS_DICT[llm_name]["full"])
    model_url = MODELS_DICT[llm_name].get("gguf_large", 
                                        MODELS_DICT[llm_name]["gguf"])
    tokenizer_name = MODELS_DICT[llm_name]["full"]

    print('Loading embeding model')
    embed_model = HuggingFaceEmbedding(model_name=config.hf_embed_model, embed_batch_size=32)#, token=os.getenv('HF_API_KEY'))
    Settings.embed_model = embed_model
    if embed_model_only:
        return {'embed_model': embed_model}
                
    print("Wrapping LLM")
    llm = LLM_WRAPPER(
                    model_name= None if not use_cuda else model_name,
                    tokenizer_name=tokenizer_name,
                    model_url = None if use_cuda else model_url,
                    tokenizer_kwargs= config.tokenizer_config,
                    generate_kwargs = config.generation_config if use_cuda else config.llamacpp_generation_config,
                    model_kwargs = {**config.model_config, "trust_remote_code": True} if use_cuda else config.llamacpp_model_config,
                            )
    Settings.llm = llm
    dspy.settings.configure(lm=llm)   
    return {'llm': llm, 'embed_model': embed_model}
