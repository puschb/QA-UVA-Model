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

documents = build_documents('all_uva_news_articles_individual')
nodes = build_nodes(documents,512,128)


for node in nodes:
  with open(f'node_storage/{node.id_}', "w", encoding='utf-8') as f:
    f.write(node.json())