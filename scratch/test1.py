import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
import gc

# load documents
documents = SimpleDirectoryReader("all_uva_news_articles").load_data()


# setup prompts - specific to StableLM
from llama_index.prompts.prompts import SimpleInputPrompt

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")


import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.cuda.mem_get_info())

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


print(torch.cuda.mem_get_info())
embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {'device': 'cpu'})
)


print(torch.cuda.mem_get_info())
print("before service context------------------------")
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model=embed_model)
print(torch.cuda.mem_get_info())
print("after service context------------------------")
index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True )
index.storage_context.persist()
'''print(torch.cuda.mem_get_info())
# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
print(torch.cuda.mem_get_info())
del embed_model
gc.collect()

response = query_engine.query("What is the new Alzheimer research that UVA Health is doing?")


print(response)'''
print('done')