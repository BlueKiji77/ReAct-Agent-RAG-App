from setuptools import setup, find_packages

setup(
    name="react_agent",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        "llama-index",
        "groq",
        "llama_deploy",
        "qdrant_client",
        "llama-index-llms-groq",
        "accelerate",
        # "flash_attn",
        "optimum",
        "auto-gptq",
        "pyvis",
        "llama-index-callbacks-langfuse",
        "arize-phoenix",
        "langfuse==2.51.5",
        "dspy-ai",
        "llama-index-llms-llama-cpp",
        "llama-index-llms-groq",
        "gradio",
        "llama-index-embeddings-huggingface"
    ],
)