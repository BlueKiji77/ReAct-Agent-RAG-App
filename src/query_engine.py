
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle
from typing import List

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.llms import LLM
from dataclasses import dataclass
from typing import Optional, List, Any
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle

from typing import List, Any, Dict
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode, Document
from llama_index.core import VectorStoreIndex
import nest_asyncio
nest_asyncio.apply()

class CustomQueryFusionRetriever(QueryFusionRetriever):
    """
    Custom implementation of QueryFusionRetriever that handles list responses from LLM.
    """
    
    def _get_queries(self, original_query: str) -> List[QueryBundle]:
        """
        Generate multiple search queries based on a single input query.
        Overrides parent method to handle list responses from LLM.
        
        Args:
            original_query (str): The initial query to expand upon
            
        Returns:
            List[QueryBundle]: List of generated query bundles
        """
        prompt_str = self.query_gen_prompt.format(
            num_queries=self.num_queries - 1,
            query=original_query,
        )
        response = self._llm.complete(prompt_str)
        
        # Handle both single response and list of responses
        if isinstance(response, list):
            print(f"response: {response}")
            queries = response[0]['text'].split("\n")
        else:
            queries = response.text.split("\n")
            
        queries = [q.strip() for q in queries if q.strip()]
        if self._verbose:
            queries_str = "\n".join(queries)
            print(f"Generated queries:\n{queries_str}")

        # The LLM often returns more queries than we asked for, so trim the list
        return [QueryBundle(q) for q in queries[: self.num_queries - 1]]



class HybridRetriver():
    def __init__(
        self,
        index: VectorStoreIndex,
        nodes: List[TextNode] = None,
#         chunk_size: int = 256,
        mode: str = "reciprocal_rerank",
        vector_similarity_top_k: int = 2,
        bm25_similarity_top_k: int = 2,
        fusion_similarity_top_k: int = 2,
        num_queries: int = 4,
        documents: List[Document] = None,
        cache_dir: str = None,
        llm = None,
        **kwargs: Any,               
        ) -> None:
        
        self.semantic_retriever = VectorIndexRetriever(index=index, similarity_top_k=vector_similarity_top_k)
        self.bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=bm25_similarity_top_k)
        self.fusion_retriever = CustomQueryFusionRetriever(
                                    retrievers=[self.semantic_retriever, self.bm25_retriever],
                                    similarity_top_k=fusion_similarity_top_k, 
                                    mode=mode,
                                    llm=Settings.llm if llm is None else llm,
                                    use_async=False
                                        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "vector_retriever": self.vector_retriever,
            "bm25_retriever": self.bm25_retriever,
            "fusion_retriever": self.fusion_retriever,
        }

    def retrieve(self, query_str: str) -> Any:
        """Retrieve."""
        return self.fusion_retriever.retrieve(query_str)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)


@dataclass
class Response:
    response: Any
    source_nodes: Optional[List] = None

    def __str__(self):
        return self.response


class StuffedContextQueryEngine:
    """My query engine.

    Uses the tree summarize response synthesis module by default.

    """

    def __init__(
        self,
        retriever: BaseRetriever,
        qa_prompt: PromptTemplate,
        llm: None,
        node_postprocessors: Any,
#         num_children=10
    ) -> None:
        self._retriever = retriever
        self._qa_prompt = qa_prompt
        self._llm = Settings.llm if llm is None else llm
        self._node_postprocessors = node_postprocessors
        
    def _apply_node_postprocessors(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes
    

    ##############
    def combine_results(self,
        texts,
        query_str,
        cur_prompt_list,
    ):
#         print('\nIN combine_results')
        context_window = self._llm.metadata.context_window
        new_texts = []
        idx = 0

        while idx <= len(texts):
            text_batch = []
            current_token_count = 0
            
            # Add texts to the batch until the context window is nearly full
            while idx < len(texts) and current_token_count + len(self._llm._tokenizer.tokenize(texts[idx])) + 2 <= context_window:
                curr_node_token_count = len(self._llm._tokenizer.tokenize(texts[idx])) + 2
                text_batch.append(texts[idx])
                node_length = curr_node_token_count
                current_token_count += curr_node_token_count
                idx += 1

            if text_batch:
                context_str = "\n\n".join(text_batch)
                fmt_qa_prompt = self._qa_prompt.format(
                    context_str=context_str, query_str=query_str
                )
#                 print(f"Combining response using {fmt_qa_prompt}")
                combined_response = self._llm.complete(fmt_qa_prompt)
#                 print(f"Combined response is: {combined_response}")
                new_texts.append(combined_response)
                cur_prompt_list.append(fmt_qa_prompt)

        if len(new_texts) == 1:
            return new_texts[0]
        else:
            return self.combine_results(
                new_texts, query_str, cur_prompt_list
            )
    
################

    def generate_response_hs(self,
        retrieved_nodes, query_str
    ):
        """Generate a response using hierarchical summarization strategy.

        Combine num_children nodes hierarchically until we get one root node.

        """
#         print('\nIN generate_response_hs')
        fmt_prompts = []
        node_responses = []
#         print(f"Retrieved node: {retrieved_nodes}")
        retrieved_nodes = [] if len(retrieved_nodes) == 0 else retrieved_nodes
        
        node_idx = 0
        stuffed_window_batch = []
        context_length = self._llm.metadata.context_window
        if len(retrieved_nodes) == 0:
            stuffed_window_batch.append("")
        while node_idx < len(retrieved_nodes):
            token_count = 0
            text_batch = []
            print(self._llm.tokenizer)
            while node_idx < len(retrieved_nodes) and (token_count + len(self._llm._tokenizer.tokenize(retrieved_nodes[node_idx].get_content())) + 2) <= context_length:
                curr_node_token_count = len(self._llm._tokenizer.tokenize(retrieved_nodes[node_idx].get_content())) + 2
#                 print(f"\nlength text to stuff: {curr_node_token_count}")
                text_batch.append(retrieved_nodes[node_idx].get_content())
                token_count += curr_node_token_count
                node_idx += 1
#                 print(f"token count is: {token_count}, with context {context_length} at node {node_idx}")
            
            window_content = "\n\n".join(text_batch)
            
#             print(f"\nstuffed window: {window_content}")
            
            stuffed_window_batch.append(window_content)
        
#         print(f"stuffed_window_batch: {stuffed_window_batch}")
#         print("==="*40)
#         print('In loop')
        for context_str in stuffed_window_batch:
#             print("==="*30)
#             context_str = node.get_content()
            fmt_qa_prompt = self._qa_prompt.format(
                context_str=context_str, query_str=query_str
            )
            # print(f"Prompt going to .complete: \n{fmt_qa_prompt}")
            node_response = self._llm.complete(fmt_qa_prompt)
            node_responses.append(node_response)
            fmt_prompts.append(fmt_qa_prompt)
        
#         print(f"Combining results")
#         print(f"Node response {node_responses}")
        
        if len(node_responses) == 1:
            if isinstance(node_responses[0], CompletionResponse):
                return (node_responses[0].text, fmt_prompts)
            else:
                return (node_responses[0][0]['text'] , fmt_prompts)
        else:
            response_txt = self.combine_results(
                [r[0]['text'] if isinstance(r, list) else r.text for r in node_responses],
                query_str,
                fmt_prompts,
            )
#             print(f"response_txt: {type(response_txt)} \n{response_txt}")
            return response_txt, fmt_prompts


    
    def query(self, query: str):
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)

        retrieved_nodes = self._retriever.retrieve(query)
        print(f"Retrieved {len(retrieved_nodes)} nodes.")

        # for node in retrieved_nodes:
        #     print(f"Node \n: {node}")
        
        retrieved_nodes = self._apply_node_postprocessors(nodes=retrieved_nodes, 
                                                         query_bundle=query_bundle)
        print(f"Left with {len(retrieved_nodes)} nodes after postprocessing.")
        
        # print(f"Retrieved source nodes: {retrieved_nodes}")
        # print(f"self._llm: {self._llm}")
        print(f"Query: {query}")
        response_txt, _ = self.generate_response_hs(
            retrieved_nodes,
            query,
        )
        response = Response(response_txt, source_nodes=retrieved_nodes)
        return response

    async def aquery(self, query: str):
        retrieved_nodes = await self._retriever.aretrieve(query)
        response_txt, _ = await agenerate_response_hs(
            retrieved_nodes,
            query,
        )
        response = Response(response_txt, source_nodes=retrieved_nodes)
        return response
