from src.preprocessing.base import BasePreprocessing

AUTHOR_ID_TOKEN = "author_id_token"

class AuthorIDReplacer(BasePreprocessing):

    def opt(self, input: list[list[str]]) -> list[str]:
        for record in input:
            result = []
            for token in record:
                if (len(token) == 32 and
                    len(set(token)) > 9): # a bit of conservativity is good.
                    token = AUTHOR_ID_TOKEN
                result.append(token)
            yield result

    @classmethod
    def short_name(cls) -> str:
        return "idr"

    def name(self) -> str:
        return "author id replacer"
