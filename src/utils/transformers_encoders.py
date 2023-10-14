from sentence_transformers import SentenceTransformer

import torch
import numpy as np
from gensim.models import KeyedVectors

import logging


logger = logging.getLogger()


class TransformersEmbeddingEncoder:

    def __init__(self, device="cpu", transformer_identifier="sentence-transformers/all-distilroberta-v1", special_token=[], *args, **kwargs):
        self.device = device
        logger.info(f"transformer embedding encoder identifier: {transformer_identifier}")
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


class Word2VecEmbeddingEncoder:

    def __init__(self, embedding_path, device="cpu") -> None:
        self.device = device
        self.embedding_path = embedding_path
        self.__word2vec__ = KeyedVectors.load_word2vec_format(self.embedding_path, binary=True)
        self.__default_vector__ = torch.zeros(size=(1, 300), device=self.device) # In case you use a different encoder or embedding, change it accordingly
        self.__zero_vector__ = self.__default_vector__

    def get_vectors(self, tokens):
        if len(tokens) == 0:
            return self.__default_vector__
        vectors = [None]*len(tokens)
        for i, t in enumerate(tokens):
            try:
                vectors[i] = self.__word2vec__.get_vector(t)
            except KeyError:
                vectors[i] = np.zeros((1, 300), dtype=np.float32)
        return torch.tensor(np.vstack(vectors), device=self.device)

    def transform(self, record):
        
        results = self.get_vectors(record)
        if len(results) == 0:
            return (self.__zero_vector__(),)
        results = results.sum(dim=0).div(len(results))        
        return (results,)

    def fit(self, *args, **kwargs):
        pass


class Word2VecEmbeddingEncoderWithContext(Word2VecEmbeddingEncoder):
    
    def get_zero_vector(self):
        return torch.zeros(300+self.context_length, device=self.device)
    
    def __init__(self, context_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_length = context_length
    
    def transform(self, record):
        if len(record[1][0]) == 0:
            return self.get_zero_vector()
        
        result = torch.cat((torch.tensor(record[0], device=self.device), super().transform(record[1][0])[0]))
        return result


class SequentialWord2VecEmbeddingEncoder(Word2VecEmbeddingEncoder):
    def transform(self, record):
        result = [None]*len(record)
        for i, sequence_records in enumerate(record):
            result[i] = super().transform(sequence_records)

        return result


class SequentialTransformersWord2VecEncoderWithContext(Word2VecEmbeddingEncoder):
    
    def __init__(self, context_length, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context_length = context_length
    
    def transform(self, record):
        try:
            result = [None]*len(record[-1])
        except:
            result = []

        contexts = [context for context in zip(*record[0])]
        for i, (context, sequence_records) in enumerate(zip(contexts, record[1])):
            temp = torch.cat((torch.tensor(context, device=self.device), super().transform(sequence_records)[0]))
            result[i] = temp
        
        if len(result) == 0:
            return ((self.get_zero_vector(),),)
        return result


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


class TransformersEmbeddingEncoderWithContext(TransformersEmbeddingEncoder):
    
    def get_zero_vector(self):
        return torch.zeros(768+self.context_length, device=self.device)
    
    def __init__(self, context_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_length = context_length
    
    def transform(self, record):
        if len(record[1][0]) == 0:
            return self.get_zero_vector()
        
        result = torch.cat((torch.tensor(record[0], device=self.device), super().transform(record[1][0])[0]))
        return result


class SequentialTransformersEmbeddingEncoderWithContext(TransformersEmbeddingEncoder):
    
    def __init__(self, context_length, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context_length = context_length
    
    def transform(self, record):
        try:
            result = [None]*len(record[-1])
        except:
            result = []

        contexts = [context for context in zip(*record[0])]
        for i, (context, sequence_records) in enumerate(zip(contexts, record[1])):
            temp = torch.cat((torch.tensor(context, device=self.device), super().transform(sequence_records)[0]))
            result[i] = temp
        
        if len(result) == 0:
            return ((self.get_zero_vector(),),)
        return result
