import re
from src.preprocessing.base import BasePreprocessing

class RepetitionRemoving(BasePreprocessing):

    def opt(self, input):
        for record in input:
            result = []
            for token in record:
                for instance, letter in re.findall(r'((\w)\2{2,})', token):
                    token = token.replace(instance, letter)
                result.append(token)
            yield result
    
    def name(self) -> str:
        return "repetition remover"
    
    @classmethod
    def short_name(cls) -> str:
        return "rr"
    