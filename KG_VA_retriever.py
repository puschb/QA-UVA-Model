try:
    from llama_index import QueryBundle
    from llama_index.indices.query.schema import QueryBundle
    from llama_index.schema import NodeWithScore
    from llama_index.retrievers import BaseRetriever
    from typing import List

    class KG_VA_retreiver(BaseRetriever):
        def __init__(self, vector_retriever, kg_retriever, mode='AND'):
            self._vector_retriever = vector_retriever
            self._kg_retriever = kg_retriever
            if mode not in ("AND", "OR"):
                raise ValueError("Invalid mode.")
            self._mode = mode

        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            kg_nodes = self._kg_retriever.retrieve(query_bundle)

            vector_ids = {n.node.node_id for n in vector_nodes}
            kg_ids = {n.node.node_id for n in kg_nodes}

            combined_dict = {n.node.node_id: n for n in vector_nodes}
            combined_dict.update({n.node.node_id: n for n in kg_nodes})

            if self._mode == "AND":
                retrieve_ids = vector_ids.intersection(kg_ids)
            else:
                retrieve_ids = vector_ids.union(kg_ids)
            
            retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
            return retrieve_nodes
        
    if __name__ == 'main':
        #build service context
        #build service context for querying
        from llama_index.llms import HuggingFaceLLM
        from llama_index.prompts.prompts import SimpleInputPrompt
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
        from llama_index import LangchainEmbedding, ServiceContext
        from llama_index.response_synthesizers import get_response_synthesizer
        from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
        - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
        - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
        - StableLM will refuse to participate in anything that could harm a human.
        """

        # This will wrap the default prompts that are internal to llama-index
        query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")


        query_llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.7, "do_sample": False},
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
            model_name="StabilityAI/stablelm-tuned-alpha-3b",
            device_map="auto",
            stopping_ids=[50278, 50279, 50277, 1, 0],
            tokenizer_kwargs={"max_length": 4096},
            model_kwargs= {'offload_folder': "offload"}
            # uncomment this if using CUDA to reduce memory usage
            #model_kwargs={"torch_dtype": torch.float16}
        )


        embed_model = LangchainEmbedding(
          HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {'device': 'cpu'})
        )
        service_context = ServiceContext.from_defaults(chunk_size=1024,llm=query_llm,embed_model=embed_model)


        import faiss
        from llama_index.vector_stores import FaissVectorStore
        from llama_index import ServiceContext, StorageContext, load_index_from_storage, load_graph_from_storage
        from llama_index.graph_stores import SimpleGraphStore
        from KG_VA_retriever import KG_VA_retreiver
        from llama_index.retrievers import VectorIndexRetriever, KGTableRetriever
        from llama_index import MockEmbedding
        from llama_index.llms import MockLLM

        #load_service_context = ServiceContext.from_defaults(llm=MockLLM(),embed_model=MockEmbedding(768))

        #load indicies from storage
        #build faiss index - using just cpu for now, will change
        faiss_vector_store = FaissVectorStore.from_persist_dir("./faiss_vector_store")
        storage_context = StorageContext.from_defaults(
            vector_store=faiss_vector_store, persist_dir="./faiss_vector_store"
        )
        fiass_index = load_index_from_storage(storage_context=storage_context, service_context=service_context)

        #build kg index
        graph_store = SimpleGraphStore.from_persist_dir("./knowledge_graph_store")
        storage_context = StorageContext.from_defaults(graph_store=graph_store,
                                                       persist_dir='./knowledge_graph_store')
        kg_graph = load_index_from_storage(storage_context=storage_context, service_context = service_context)



        #create retrievers
        v_retriever = VectorIndexRetriever(
            index=fiass_index,
            similarity_top_k=3,
            vector_store_query_mode="default",
            alpha=None,
            doc_ids=None,
        ) #https://gpt-index.readthedocs.io/en/latest/core_modules/data_modules/index/vector_store_guide.html

        kg_retriever = KGTableRetriever(
            index=kg_graph, 
            retriever_mode="hybrid", 
            include_text=True, 
            similarity_top_k= 3,
            use_global_node_triplets=True
        )

        custom_retriever = KG_VA_retreiver(v_retriever,kg_retriever, mode='OR')

        
        response_synthesizer = get_response_synthesizer(response_mode='compact', service_context=service_context)


        #build query engine
        query_engine = RetrieverQueryEngine(retriever = custom_retriever,response_synthesizer=response_synthesizer)

        response = query_engine.query("What McIntire student wrote for Key&Peele?")

        print(response)
except Exception as e:
    print(e)
