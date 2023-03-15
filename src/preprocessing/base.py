from src.utils.commons import RegisterableObject

class BasePreprocessing(RegisterableObject):

    def opt(self, input):
        pass

    def name(self) -> str:
        raise NotImplemented()

    def short_name(self) -> str: # At most 2 chars
        raise NotImplemented
    