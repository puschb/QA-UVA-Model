from llama_index import StorageContext, load_index_from_storage
import logging
import sys
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index import ServiceContext, LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import json
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import os



logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    #generate_kwargs = {'device': 'cpu'}
)



embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {'device': 'cpu'})
)
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model=embed_model)

print("done loading models")
print("done")


# Test on some sample data
json_value = {
    "blogPosts": [
        {"id": 1, "title": "First blog post", "content": "This is my first blog post"},
        {
            "id": 2,
            "title": "Second blog post",
            "content": "This is my second blog post"
        }
    ],
    "comments": [
        {"id": 1, "content": "Nice post!", "username": "jerry", "blogPostId": 1},
        {
            "id": 2,
            "content": "Interesting thoughts",
            "username": "simon",
            "blogPostId": 2
        },
        {
            "id": 3,
            "content": "Loved reading this!",
            "username": "simon",
            "blogPostId": 2
        }
    ]
}

# JSON Schema object that the above JSON value conforms to
json_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "description": "Schema for a very simple blog post app",
    "type": "object",
    "properties": {
        "blogPosts": {
            "description": "List of blog posts",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "description": "Unique identifier for the blog post",
                        "type": "integer"
                    },
                    "title": {
                        "description": "Title of the blog post",
                        "type": "string"
                    },
                    "content": {
                        "description": "Content of the blog post",
                        "type": "string"
                    }
                },
                "required": ["id", "title", "content"]
            }
        },
        "comments": {
            "description": "List of comments on blog posts",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "description": "Unique identifier for the comment",
                        "type": "integer"
                    },
                    "content": {
                        "description": "Content of the comment",
                        "type": "string"
                    },
                    "username": {
                        "description": "Username of the commenter (lowercased)",
                        "type": "string"
                    },
                    "blogPostId": {
                        "description": "Identifier for the blog post to which the comment belongs",
                        "type": "integer"
                    }
                },
                "required": ["id", "content", "username", "blogPostId"]
            }
        }
    },
    "required": ["blogPosts", "comments"]
}

print("done")



from llama_index.indices.service_context import ServiceContext
from llama_index.llms import OpenAI
from llama_index.indices.struct_store import JSONQueryEngine


nl_query_engine = JSONQueryEngine(
    json_value=json_value, json_schema=json_schema, service_context=service_context,verbose=True
)
raw_query_engine = JSONQueryEngine(
    json_value=json_value,
    json_schema=json_schema,
    service_context=service_context,
    synthesize_response=False,
)

print("done")

nl_response = nl_query_engine.query(
    "What comments has Jerry been writing?",
)

print(nl_response)
