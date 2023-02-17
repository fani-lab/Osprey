import nltk

from nltk.corpus import stopwords


from preprocessing.base import BasePreprocessing


class NLTKStopWordRemoving(BasePreprocessing):
    
    def __init__(self) -> None:
        super(NLTKStopWordRemoving).__init__()
        nltk.download('stopwords')
        

    def opt(self, input: list[list[str]]) -> list[str]:
        # super(self, NLTKStopWordRemoving).opt(input)
        sw_set = stopwords.words()
        for record in input:
            result = []
            for token in record:
                if token not in sw_set:
                    result.append(token)
            yield result
        # return input.apply(lambda record: [token for token in record if token not in sw_set], axis=1)


