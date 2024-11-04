# src/config.py
import os
import torch
from dotenv import load_dotenv
from pathlib import Path

class Config:
    def __init__(self, env_file='config.env'):
        # Get the project root directory (parent of src)
        root_dir = Path(__file__).parent.parent
        
        # Try multiple possible env files
        possible_env_files = ['config.env', '.env', 'settings.env']
        
        env_loaded = False
        for env_file in possible_env_files:
            # Create full path by joining root directory with env filename
            env_path = root_dir / env_file
            if env_path.exists():
                load_dotenv(env_path)
                env_loaded = True
                print(f"Loaded environment from {env_path}")
                break
                
        if not env_loaded:
            print("Warning: No environment file found. Using system environment variables.")
        
        # API Keys
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.langfuse_secret_keey = os.getenv('LANGFUSE_SECRET_KEY')
        self.langfuse_public_keey = os.getenv('LANGFUSE_PUBLIC_KEY')
        self.langfuse_host = os.getenv('LANGFUSE_HOST')
        self.tavily_key = os.getenv('TAVILY_KEY')
        self.hf_api_key = os.getenv('HF_API_KEY')

        # Models names
        self.hf_tiny_model = os.getenv('HF_TINY_MODEL_NAME')
        self.hf_small_model = os.getenv('HF_SMALL_MODEL_NAME')
        self.hf_embed_model = os.getenv('HF_EMBEDDING_MODEL')
        self.hf_big_model = os.getenv('HF_BIG_MODEL_NAME')
        self.groq_small_model = os.getenv('GROQ_SMALL_MODEL_NAME')
        self.groq_big_model = os.getenv('GROQ_BIG_MODEL_NAME')

        # Model Config
        self.tokenizer_config = dict(
                                    max_length=512*16,
                                    )
        self.model_config = dict(
                                max_length=512*16,
                                torch_dtype=torch.float16,
                                trust_remote_code=True,
                                device_map="balanced"
                                )
        self.generation_config = dict( 
                                    temperature=0.5, 
                                    do_sample=True,
                                    max_new_tokens=512*2,
                                    add_gen_prompt=True,
                                    top_k=40,
                                    top_p=0.95,
                                    repetition_penalty=1.1
                                  )
        self.llamacpp_model_config = dict(
        #                                **MODEL_CONFIG, 
                                        n_gpu_layers = 1,
        #                                 max_length = TOKENIER_CONFIG["max_length"], 
        #                                 verbose = True
                                        )
        self.llamacpp_generation_config = dict(
                                            temperature=0.5,  
                                            max_new_tokens=512*2,
                                            add_gen_prompt=True
                                             )
        
        # Validate required keys are present
        self._validate_config()
    
    def _validate_config(self):
        required_vars = [
            ('GROQ_API_KEY', self.groq_api_key),
            ('LANGFUSE_SECRET_KEY', self.langfuse_secret_keey),
            ('LANGFUSE_PUBLIC_KEY', self.langfuse_public_keey),
            ('LANGFUSE_HOST', self.langfuse_host),
            ('TAVILY_KEY', self.tavily_key),
            ('HF_API_KEY', self.hf_api_key)
        ]
        
        missing_vars = [var_name for var_name, var_value in required_vars if not var_value]
        
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                "Please ensure all required variables are set in your environment file."
            )

# Usage
# config = Config()