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
    


