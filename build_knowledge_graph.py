
from transformers import pipeline

triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large', device='cuda')


#function to extract kg triplets from data
#code from Babelscape/rebel-large model card
#https://huggingface.co/Babelscape/rebel-large

def extract_triplets(input_text):
    text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(input_text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])[0]

    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append((subject.strip(), relation.strip(), object_.strip()))

    return triplets


from llama_index import Document
import os
import json
def build_documents(directory):
  documents = []
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
  return documents


from llama_index.node_parser import SimpleNodeParser
def build_nodes(documents, chunk_size, chunk_overlap):
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = node_parser.get_nodes_from_documents(documents=documents, show_progress=True)
    return nodes


from llama_index import ServiceContext, MockEmbedding
from llama_index.llms import MockLLM
#need to provide these otherwise it will require and openai key
llm = MockLLM()
embed_model = MockEmbedding(embed_dim=768)

service_context = ServiceContext.from_defaults(chunk_size=512,llm=llm,embed_model=embed_model)


#build docs and nodes
documents = build_documents('all_uva_news_articles_individual')
nodes = build_nodes(documents,512,128)


#build knowledge graph index
from llama_index import KnowledgeGraphIndex, StorageContext
from llama_index.graph_stores import SimpleGraphStore

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)
index = KnowledgeGraphIndex(nodes=nodes, kg_triplet_extract_fn=extract_triplets, 
                            service_context=service_context, storage_context=storage_context, show_progress=True,
                            max_triplets_per_chunk=10)

index.storage_context.persist('./knowledge_graph_store')
