from typing import Any, List
from InstructorEmbedding import INSTRUCTOR
from llama_index.embeddings.base import BaseEmbedding

class HuggingfaceAPIEmbedding(BaseEmbedding):
  def __init__(
    self, 
    instructor_model_name: str = "hkunlp/instructor-large",
    instruction: str = "Represent the Computer Science documentation or question:",
    **kwargs: Any,
  ) -> None:
    self._model = INSTRUCTOR(instructor_model_name)
    self._instruction = instruction
    super().__init__(**kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
      embeddings = self._model.encode([[self._instruction, query]])
      return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
      embeddings = self._model.encode([[self._instruction, text]])
      return embeddings[0] 

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
      embeddings = self._model.encode([[self._instruction, text] for text in texts])
      return embeddings