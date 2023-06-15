from torch import sparse_coo_tensor, float32
import numpy as np

from heapq import nlargest

NUMBER_OF_PREDEFINED_VECTORS = 1 # Should be handled dynamically later. needs to synced with the context vector

class OneHotEncoder:

    def __init__(self, buffer_cap=20, device="cpu", vector_size=-1, vectors_dimensions=2) -> None:
        
        self.__buffer_cap = buffer_cap
        self.vector_size = vector_size
        self.records = dict()
        self.device = device
        self.transform_started = False
        self.vectors_dimension = [1] * vectors_dimensions
        self.vectors = dict()
        self.__default_vector = None
        self.__zero_vector = None
    
    def create_sparse_vector(self, index):
        if index is None:
            ones = [[] for i in range(len(self.vectors_dimension))]
        else:
            ones = [(0,) if i!=1 else (index,) for i in range(len(self.vectors_dimension))]
        return sparse_coo_tensor(ones, (1.0,)*(index != None), size=self.vectors_dimension, dtype=float32)

    def generate_sparse_vectors(self):
        for i, (k, _) in enumerate(self.records.items()):
            self.vectors[k] = self.create_sparse_vector(i)
        
        del self.records
    
    def __get_default_vector(self):
        if self.__default_vector is None:
            self.__default_vector = self.create_sparse_vector(self.vectors_dimension[1]-1)
        return self.__default_vector

    def __get_zero_vector(self):
        if self.__zero_vector is None:
            self.__zero_vector = self.create_sparse_vector(None)
        return self.__zero_vector

    def get_number_of_predefined_vectors(self):
        return 1 # the `1` is for the default vector; look at `__get_default_vector`
    
    def transform(self, record):
        if not self.transform_started:
            self.generate_sparse_vectors()
        self.transform_started = True
        default_vector = self.__get_default_vector()
        
        result = []
        for token in record:
            result.append(self.vectors.get(token, default_vector))
        
        if len(result) == 0:
            return (self.get_zero_vector(),)
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
            mostfrequent = nlargest(self.vector_size - self.get_number_of_predefined_vectors(), all_tokens_count, key=lambda x:x[1])
            self.records = {entry[0]: entry[1] for entry in mostfrequent}

        self.vectors_dimension[1] = len(self.records) + self.get_number_of_predefined_vectors()

    get_zero_vector = __get_zero_vector


class SequentialOneHotEncoder(OneHotEncoder):

    def transform(self, record):
        result = []
        for sequence_records in record:
            result.append(super().transform(sequence_records))
        
        if len(result) == 0:
            return ((self.get_zero_vector(),),)
        return result


class SequentialOneHotEncoderWithContext(OneHotEncoder):
    
    def __init__(self, context_length, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context_length = context_length
        
    def get_number_of_predefined_vectors(self):
        return super().get_number_of_predefined_vectors() + self.context_length
    
    def generate_sparse_vectors(self):
        for i, (k, _) in enumerate(self.records.items(), start=self.context_length): # we force the context features to be at the first of the feature vector
            self.vectors[k] = self.create_sparse_vector(i)
        
        del self.records
    
    def transform(self, record):
        try:
            result = [None]*len(record[-1])
        except:
            result = []
        get_context_properly_for_transformation = lambda x: x

        if self.context_length == 1:
            get_context_properly_for_transformation = lambda x: [x,]
        dimensions_of_context = ((0,) * self.context_length, tuple(range(0, self.context_length)))
        contexts = [context for context in zip(*record[0])]
        for i, (context, sequence_records) in enumerate(zip(contexts, record[1])):
            if len(sequence_records) == 0:
                result[i] = (self.get_zero_vector(),)
                continue
            temp = super().transform(sequence_records) + \
                [sparse_coo_tensor(dimensions_of_context, get_context_properly_for_transformation(context), size=self.vectors_dimension, dtype=float32)]
            result[i] = temp
        
        if len(result) == 0:
            return ((self.get_zero_vector(),),)
        return result
