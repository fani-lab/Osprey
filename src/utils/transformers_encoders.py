from sentence_transformers import SentenceTransformer




model = SentenceTransformer


class TransformersEmbeddingEncoder:

    def __init__(self, device="cpu", transformer_identifier="sentence-transformers/all-distilroberta-v1", *args, **kwargs):
        self.device = device
        self.encoder = SentenceTransformer(transformer_identifier, device=device)
    
    def transform(self, record):

        result = self.encoder.encode(" ".join(record), convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)

        return (result,) # For the consistency of the transform return value


    def fit(self, *args, **kwargs):
        pass