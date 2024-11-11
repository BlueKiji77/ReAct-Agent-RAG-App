from src import Config
from llama_deploy import deploy_workflow, deploy_core, WorkflowServiceConfig, ControlPlaneConfig, SimpleMessageQueueConfig
from src import ReActAgent
from src.utils import download_10k_reports
from src.utils import load_hf_model
from src.utils import initialize_groq_model
from src.ingest import setup_indices
from llama_index.core import PromptTemplate
from src.query_engine import HybridRetriver, StuffedContextQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from dataclasses import dataclass
from typing import Optional
import argparse
from dotenv import load_dotenv

@dataclass
class DeploymentConfig:
    local_model: bool = False
    host: str = "localhost"
    port: int = 8000
    service_name: str = "react_workflow"

def parse_arguments() -> DeploymentConfig:
    """Parse command line arguments and return DeploymentConfig."""
    parser = argparse.ArgumentParser(description='Deploy the workflow service with custom configuration.')
    
    parser.add_argument('--local-model', 
                       action='store_true',
                       help='Use local model instead of Groq')
    
    parser.add_argument('--host',
                       type=str,
                       default="localhost",
                       help='Host address for the service (default: localhost)')
    
    parser.add_argument('--port',
                       type=int,
                       default=8000,
                       help='Port number for the service (default: 8000)')
    
    parser.add_argument('--service-name',
                       type=str,
                       default="react_worflow",
                       help='Name of the service (default: my_workflow)')
    
    args = parser.parse_args()
    
    return DeploymentConfig(
        local_model=args.local_model,
        host=args.host,
        port=args.port,
        service_name=args.service_name
    )

async def server(deployment_config: Optional[DeploymentConfig] = None) -> None:
    """
    Main function to set up and deploy the workflow.
    
    Args:
        deployment_config (Optional[DeploymentConfig]): Configuration for deployment.
            If None, default values will be used.
    """
    if deployment_config is None:
        deployment_config = DeploymentConfig()

    download_10k_reports()
    config = Config()
    
    # Model initialization based on local_model flag
    if deployment_config.local_model:
        hf_models = load_hf_model(config.hf_tiny_model, config.hf_embed_model)
    else:
        hf_models = load_hf_model(config.hf_tiny_model, config.hf_embed_model, embed_model_only=True)
        groq_llama_8b = initialize_groq_model("llama3-8b-8192")
        groq_llama_70b = initialize_groq_model("llama3-70b-8192")
    
    indices = setup_indices(hf_models['embed_model'])
    lyft_index = indices['lyft_index']
    uber_index = indices['uber_index']
    
    lyft_retriever = VectorIndexRetriever(index=lyft_index, similarity_top_k=20)
    uber_retriever = VectorIndexRetriever(index=uber_index, similarity_top_k=20)
    
    colbert_reranker = ColbertRerank(
        top_n=10,
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        keep_retrieval_score=True,
    )
    
    postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7),
                    colbert_reranker]                       
        
    qa_prompt = PromptTemplate(
        """\
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query_str}
    Answer: \
    """
    )
    
    lyft_query_engine = StuffedContextQueryEngine(
                        retriever=lyft_retriever, 
                        qa_prompt=qa_prompt,
                        llm=groq_llama_8b,
                        node_postprocessors=postprocessors
                        )
    
    uber_query_engine = StuffedContextQueryEngine(
                        retriever=uber_retriever, 
                        qa_prompt=qa_prompt,
                        llm=groq_llama_8b,
                        node_postprocessors=postprocessors
                        )
    
    query_engine_tools = [
        QueryEngineTool(
            query_engine=lyft_query_engine,
            metadata=ToolMetadata(
                name="lyft_10k",
                description="Provides information about Lyft financials for year 2021",
            ),
        ),
        QueryEngineTool(
            query_engine=uber_query_engine,
            metadata=ToolMetadata(
                name="uber_10k",
                description="Provides information about Uber financials for year 2021",
            ),
        ),
    ]
    
    await deploy_core(
        control_plane_config=ControlPlaneConfig(
            host=deployment_config.host,
            port=deployment_config.port
        ),
        message_queue_config=SimpleMessageQueueConfig(),
    )
    
    await deploy_workflow(
        ReActAgent(
            llm=groq_llama_8b,
            tools=query_engine_tools,
            timeout=400,
            verbose=False
        ),
        WorkflowServiceConfig(
            host=deployment_config.host,
            port=deployment_config.port,
            service_name=deployment_config.service_name
        ),
        ControlPlaneConfig(
            host=deployment_config.host,
            port=deployment_config.port
        ),
    )

if __name__ == "__main__":
    import asyncio
    
    # Parse command line arguments
    config = parse_arguments()
    
    # Run with parsed config
    asyncio.run(server(config))