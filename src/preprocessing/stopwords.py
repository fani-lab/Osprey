import nltk

from nltk.corpus import stopwords


from src.preprocessing.base import BasePreprocessing


class NLTKStopWordRemoving(BasePreprocessing):
    
    def __init__(self) -> None:
        super(NLTKStopWordRemoving).__init__()
        nltk.download('stopwords')
        
    def opt(self, input: list[list[str]]) -> list[str]:
        sw_set = stopwords.words()
        for record in input:
            result = []
            for token in record:
                if token not in sw_set:
                    result.append(token)
            yield result

    def name(self) -> str:
        return "nltk stopwords remover"

    @classmethod
    def short_name(cls) -> str:
        return "sw"
