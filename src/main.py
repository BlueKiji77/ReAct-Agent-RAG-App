from src import Config
from llama_deploy import deploy_workflow, WorkflowServiceConfig, ControlPlaneConfig
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

async def main(local_model=False):
    download_10k_reports()
    config = Config()
    if local_model:
        hf_models = load_hf_model(config.hf_tiny_model, config.hf_embed_model)
    else:
        hf_models = load_hf_model(config.hf_tiny_model, config.hf_embed_model, embed_model_only=True)
        groq_llama_8b = initialize_groq_model("llama3-8b-8192")
        groq_llama_70b = initialize_groq_model("llama3-70b-8192")
    
    indices = setup_indices(hf_models['embed_model'])
    lyft_index = indices['lyft_index']
    uber_index = indices['uber_index']

    # lyft_retriever = HybridRetriver(index=lyft_index, vector_similarity_top_k=20, bm25_similarity_top_k=20, fusion_similarity_top_k=20, llm=groq_llama_8b)
    # uber_retriever = HybridRetriver(index=uber_index, vector_similarity_top_k=20, bm25_similarity_top_k=20, fusion_similarity_top_k=20, llm=groq_llama_8b)
    
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
    # await deploy_workflow(
    #     ReActAgent(
    #         llm=groq_llama_8b,
    #         tools=query_engine_tools,
    #         timeout=120,
    #         verbose=False
    #     ),
    #     WorkflowServiceConfig(host="127.0.0.1", port=8000, service_name="my_workflow"),
    #     ControlPlaneConfig(host="127.0.0.1", port=8000),
    # )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())