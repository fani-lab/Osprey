from src.preprocessing.base import BasePreprocessing


class AuthorIDReplacer(BasePreprocessing):
    AUTHOR_ID_TOKEN = "author_id_token"

    def opt(self, input: list[list[str]]) -> list[str]:
        for record in input:
            result = []
            for token in record:
                if (len(token) == 32 and
                    len(set(token)) > 9): # a bit of conservativity is good.
                    token = self.AUTHOR_ID_TOKEN
                result.append(token)
            yield result

    @classmethod
    def short_name(cls) -> str:
        return "idr"

    def name(self) -> str:
        return "author id replacer"


class AuthorIDReplacerBert(AuthorIDReplacer):
    AUTHOR_ID_TOKEN = "[unused0]"

    @classmethod
    def short_name(cls) -> str:
        return "bert_idr"

    def name(self) -> str:
        return "bert author id replacer"
