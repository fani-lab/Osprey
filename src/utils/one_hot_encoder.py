from torch import sparse_coo_tensor, float32


class OneHotEncoder:

    def __init__(self, buffer_cap=20, device="cpu") -> None:
        
        self.__buffer_cap = buffer_cap
        self.__records = dict()
        self.device = device
        self.transform_started = False
        self.__vectors_dimension = (1,1)
        self.vectors = dict()
        self.__default_vector = None
        self.__zero_vector = None
    
    def __generate_sparse_vectors(self):
        for k, v in self.__records.items():
            self.vectors[k] = sparse_coo_tensor(((0,), (v,)),(1.0), size=self.__vectors_dimension, dtype=float32)
        
        del self.__records
    
    def __get_default_vector(self):
        if self.__default_vector is None:
            self.__default_vector = sparse_coo_tensor(((0,), (self.__vectors_dimension[1]-1,)),(1.0), size=self.__vectors_dimension, dtype=float32)
        return self.__default_vector

    def __get_zero_vector(self):
        if self.__zero_vector is None:
            self.__zero_vector = sparse_coo_tensor((([], [])),[], size=self.__vectors_dimension, dtype=float32)
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
            if record not in self.__records:
                self.__records[record] = len(self.__records)
        buffer = {}

    def fit(self, data_generator):
        buffer = []
        for i, record in enumerate(data_generator()):
            if self.transform_started:
                raise Exception("cannot fit the encoder as this encoder has already transformed some records.")
            buffer.append(record) # todo
            if i % self.__buffer_cap == 0:
                self.flush_buffer(buffer)
        
        self.flush_buffer(buffer)
        self.__vectors_dimension = (1, len(self.__records) + 1)
