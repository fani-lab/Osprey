import re

from src.preprocessing.base import BasePreprocessing


class PunctuationRemoving(BasePreprocessing):

    def __init__(self) -> None:
        super().__init__()
    
    def opt(self, input: list[list[str]]) -> list[str]:
        for record in input:
            result = []
            for token in record:
                t = re.sub(r"[^A-Za-z0-9\s]+", '', token)
                if t:
                    result.append(t)
            yield result

    def name(self) -> str:
        return "punctuation remover"

    @classmethod
    def short_name(cls) -> str:
        return "pr"
    