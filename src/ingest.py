import qdrant_client
from tqdm import tqdm
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from IPython.display import Markdown, display
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.ingestion import IngestionCache, DocstoreStrategy
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.ingestion import IngestionPipeline
import os

def setup_indices(embed_model):
    store_path = "qdrant_store"
    lyft_collection_name = "lyft_filings"
    uber_collection_name = "uber_filings"
    
    lyft_persist_dir = f"{store_path}/indices_store/{lyft_collection_name}"
    uber_persist_dir = f"{store_path}/indices_store/{uber_collection_name}"
    
    # Check if indices already exist
    if os.path.exists(lyft_persist_dir) and os.path.exists(uber_persist_dir):
        print("Found existing indices, loading from disk...")
        
        client = qdrant_client.QdrantClient(path=store_path)
        
        # Set up vector stores
        lyft_qdrant_vector_store = QdrantVectorStore(
            client=client, 
            collection_name=lyft_collection_name
        )
        uber_qdrant_vector_store = QdrantVectorStore(
            client=client, 
            collection_name=uber_collection_name
        )
        
        # Load storage contexts from disk
        lyft_storage_context = StorageContext.from_defaults(
            vector_store=lyft_qdrant_vector_store,
            persist_dir=lyft_persist_dir
        )
        uber_storage_context = StorageContext.from_defaults(
            vector_store=uber_qdrant_vector_store,
            persist_dir=uber_persist_dir
        )
        
        # Load indices
        lyft_index = VectorStoreIndex.from_vector_store(
            vector_store=lyft_qdrant_vector_store,
            storage_context=lyft_storage_context
        )
        uber_index = VectorStoreIndex.from_vector_store(
            vector_store=uber_qdrant_vector_store,
            storage_context=uber_storage_context
        )
        
        return {'lyft_index': lyft_index, "uber_index": uber_index}
    
    print("No existing indices found, creating new ones...")
    # load data
    lyft_docs = SimpleDirectoryReader(
                  input_files=["src/data/10k/lyft_2021.pdf"]
                    ).load_data()
    uber_docs = SimpleDirectoryReader(
                    input_files=["src/data/10k/uber_2021.pdf"]
                ).load_data()
    
    print(f"lyft set has {len(lyft_docs)} documents")
    print(f"uber set has {len(uber_docs)} documents")
    
    parser_chk512_o80 = SentenceSplitter(chunk_size=512, chunk_overlap=80)
    
    client = qdrant_client.QdrantClient(path=store_path)
    
    lyft_qdrant_vector_store = QdrantVectorStore(client=client, collection_name=lyft_collection_name)
    uber_qdrant_vector_store = QdrantVectorStore(client=client, collection_name=uber_collection_name)
        
    lyft_docstore = SimpleDocumentStore()
    uber_docstore = SimpleDocumentStore()
    
    lyft_cache = IngestionCache()
    uber_cache = IngestionCache()
        
    lyft_pipeline = IngestionPipeline(
        transformations=[parser_chk512_o80, embed_model],
        cache=lyft_cache,
        docstore=lyft_docstore,
        vector_store=lyft_qdrant_vector_store
    )
    
    uber_pipeline = IngestionPipeline(
        transformations=[parser_chk512_o80, embed_model],
        cache=uber_cache,
        docstore=uber_docstore,
        vector_store=uber_qdrant_vector_store
    )
    
    lyft_nodes = lyft_pipeline.run(documents=lyft_docs[:])
    uber_nodes = uber_pipeline.run(documents=uber_docs[:])
    
    lyft_storage_context = StorageContext.from_defaults(
        vector_store=lyft_qdrant_vector_store,
        docstore=lyft_docstore,
        index_store=SimpleIndexStore(),
        persist_dir=lyft_persist_dir,
    )
    uber_storage_context = StorageContext.from_defaults(
        vector_store=uber_qdrant_vector_store,
        docstore=uber_docstore,
        index_store=SimpleIndexStore(),
        persist_dir=uber_persist_dir,
    )
    
    lyft_index = VectorStoreIndex(lyft_nodes, storage_context=lyft_storage_context)
    uber_index = VectorStoreIndex(uber_nodes, storage_context=uber_storage_context)
    
    lyft_index_name = f"{lyft_collection_name}_index"
    lyft_index.set_index_id(lyft_index_name)
    lyft_storage_context.persist(lyft_persist_dir)
    
    uber_index_name = f"{uber_collection_name}_index"
    uber_index.set_index_id(uber_index_name)
    uber_storage_context.persist(uber_persist_dir)

    return {'lyft_index': lyft_index, "uber_index": uber_index}