from llama_index import Document, VectorStoreIndex
import os, json
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
from llama_index import ServiceContext
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#load documents
documents = []
directory = 'all_uva_news_articles_individual'
for article in os.listdir(directory):
    article_path = os.path.join(directory,article)
    with open(article_path, "r",encoding = 'utf-8') as f:
        article = json.loads(f.read())
    content = article['text']
    article.pop('text')
    article.pop('url')

    #this is a very rare occurance (happens once as of 8/11) so this quick fix is fine
    description = article['description']
    if len(description)>350:
        article['description'] = description[:350]


    doc = Document(text=content,metadata=article)
    doc.id_ = article_path
    documents.append(doc)


print(len(documents))



#build service context
system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

#llm needs to be passed in for the embedding because of how the faiss vector store is implemented on llamaindex
llm = HuggingFaceLLM(
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
    # uncomment this if using CUDA to reduce memory usage
    #model_kwargs={"torch_dtype": torch.float16}
)


embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {'device': 'cpu'})
)

service_context = ServiceContext.from_defaults(chunk_size=1024,llm=llm,embed_model=embed_model)


#create nodes
node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=128)
nodes = node_parser.get_nodes_from_documents(documents=documents, show_progress=True)

#building vector store index
from llama_index import VectorStoreIndex
from llama_index import ServiceContext, StorageContext
import faiss #only have faiss-cpu installed for now, to get gpu:pip install faiss-gpu
from llama_index.vector_stores import FaissVectorStore
service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)


#build faiss index
d = 768 
faiss_index = faiss.IndexFlatL2(d)
faiss_vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=faiss_vector_store)
v_faiss_index = VectorStoreIndex(nodes=nodes, service_context=service_context, storage_context=storage_context,show_progress=True)
#store faiss index
v_faiss_index.storage_context.persist("./faiss_vector_store")

