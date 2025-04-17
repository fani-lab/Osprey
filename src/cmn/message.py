# from cmn.conversation import Conversation

class Message():
    def __init__(
        self,
        author_id: str,
        time: str,
        text: str,
        prev: None,
        next: None,
        conv # Converaiton Instance 
    ):

        self.author_id = author_id
        self.time = time
        self.text = text
        self.prev = prev
        self.next = next
        self.conv = conv