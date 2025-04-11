# from cmn.conversation import Conversation

class Message():
    def __init__(
        self,
        author_id: str,
        time: str,
        text: str,
        conv
    ):

        self.author_id = author_id
        self.time = time
        self.text = text
        self.conv = conv