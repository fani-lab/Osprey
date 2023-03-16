from src.utils.commons import RegisterableObject

class BasePreprocessing(RegisterableObject):

    def opt(self, input):
        pass

    def name(self) -> str:
        raise NotImplementedError()
    