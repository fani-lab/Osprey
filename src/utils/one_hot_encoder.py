from torch import sparse_coo_tensor, float32
import numpy as np

from heapq import nlargest


class OneHotEncoder:

    def __init__(self, buffer_cap=20, device="cpu", vector_size=-1, vectors_dimensions=2) -> None:
        
        self.__buffer_cap = buffer_cap
        self.vector_size = vector_size
        self.records = dict()
        self.device = device
        self.transform_started = False
        self.__vectors_dimension = [1] * vectors_dimensions
        self.vectors = dict()
        self.__default_vector = None
        self.__zero_vector = None
    
    def __create_sparse_vector(self, index):
        if index is None:
            ones = [[] for i in range(len(self.__vectors_dimension))]
        else:
            ones = [(0,) if i!=1 else (index,) for i in range(len(self.__vectors_dimension))]
        return sparse_coo_tensor(ones, (1.0,)*(index != None), size=self.__vectors_dimension, dtype=float32)

    def __generate_sparse_vectors(self):
        for i, (k, _) in enumerate(self.records.items()):
            self.vectors[k] = self.__create_sparse_vector(i)
        
        del self.records
    
    def __get_default_vector(self):
        if self.__default_vector is None:
            self.__default_vector = self.__create_sparse_vector(self.__vectors_dimension[1]-1)
        return self.__default_vector

    def __get_zero_vector(self):
        if self.__zero_vector is None:
            self.__zero_vector = self.__create_sparse_vector(None)
        return self.__zero_vector

    def transform(self, record):
        if not self.transform_started:
            self.__generate_sparse_vectors()
        self.transform_started = True
        default_vector = self.__get_default_vector()
        
        result = []
        for token in record:
            result.append(self.vectors.get(token, default_vector))
        
        if len(result) == 0:
            return (self.__get_zero_vector(),)
        return result

    def flush_buffer(self, buffer):
        for record in buffer:
            count = self.records.get(record, 0)
            self.records[record] = count + 1

    def fit(self, data_generator):
        buffer = []
        for i, record in enumerate(data_generator(), start=1):
            if self.transform_started:
                raise Exception("cannot fit the encoder as this encoder has already transformed some records.")
            buffer.append(record)
            if i % self.__buffer_cap == 0:
                self.flush_buffer(buffer)
                buffer = []
        
        self.flush_buffer(buffer)
        
        if self.vector_size > 0:
            all_tokens_count = [(k, v) for k,v in self.records.items()]
            mostfrequent = nlargest(self.vector_size, all_tokens_count, key=lambda x:x[1])
            self.records = {entry[0]: entry[1] for entry in mostfrequent}

        self.__vectors_dimension[1] = len(self.records) + 1
