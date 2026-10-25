import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self,model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self,texts:list[str])->np.ndarray:
        if not texts:
            return np.zeros((0,self.dim),dtype=np.float32)
        return self.model.encode(texts,normalize_embeddings=True,convert_to_numpy=True).astype(np.float32)

    def embed_query(self,text:str)->np.ndarray:
        return self.embed_documents([text][0]) 