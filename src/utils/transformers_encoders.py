from sentence_transformers import SentenceTransformer

import torch

import logging


logger = logging.getLogger()

class TransformersEmbeddingEncoder:

    def __init__(self, device="cpu", transformer_identifier="sentence-transformers/all-distilroberta-v1", special_token=[], *args, **kwargs):
        self.device = device
        self.encoder = SentenceTransformer(transformer_identifier, device=device)

        # we should call add_special_tokens for [unusedX] tokens, because the tokenizer consider them unkown.
        #   Though the size of the vocab won't change because [unusedX] are already there
        if len(special_token) > 0:
            self.encoder.tokenizer.add_special_tokens({"additional_special_tokens": special_token})
    
    def transform(self, record):

        result = self.encoder.encode(" ".join(record), convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)

        return (result,) # For the consistency of the transform return value

    def fit(self, *args, **kwargs):
        pass


class GloveEmbeddingEncoder:
    
    def __init__(self, embedding_path, device="cpu") -> None:
        self.device = device
        self.embedding_path = embedding_path
        self.__glove__ = None
        self.__default_vector__ = torch.zeros(size=(int(self.embedding_path.split(".")[-2].replace("d","")),))
        self.__zero_vector__ = self.__default_vector__

    def glove(self):
        if self.__glove__ is None:
            self.__glove__ = dict()
            with open(self.embedding_path, "r", encoding="utf8") as f:
                logger.info(f"loading glove encoder at {self.embedding_path}")
                for l in f.readlines():
                    word, *vec = l.split(" ")
                    self.__glove__[word] = torch.Tensor([float(num) for num in vec])
                logger.debug("done loading glove encoder")
        return self.__glove__

    def transform(self, record):
        result = [self.glove().get(token, self.__default_vector__) for token in record]

        if len(result) == 0:
            result = [self.__zero_vector__,]
        vec = torch.stack(result).sum(dim=0).div(len(result))
        
        return (vec,)

    def fit(self, *args, **kwargs):
        pass

class SequentialTransformersEmbeddingEncoder(TransformersEmbeddingEncoder):


        # result = self.encoder.encode(" ".join(record), convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)

        # return (result,) # For the consistency of the transform return value

    def transform(self, record):
        result = []
        for sequence_records in record:
            result.append(super().transform(sequence_records))
        
        # if len(result) == 0:
        #     return ((self.get_zero_vector(),),)
        return result
