import re

from preprocessing.base import BasePreprocessing


class PunctuationRemoving(BasePreprocessing):

    def __init__(self) -> None:
        super().__init__()
    
    def opt(self, input: list[list[str]]) -> list[str]:
        for record in input:
            result = []
            
            for token in record:
                try:
                    t = re.sub(r"^[\w]+$", '', token)
                    if t:
                        result.append(t)
                except:
                    print()
            yield result

