class Message(Object):
    def __init__(
        self,
        idx: str,
        author_id: str,
        time: str,
        n_chars: str,
        n_words: str,
        #conversation_size: str, should go to conversation class
        #number_of_authors: str, should go to conversation class
        text: str,
        tagged_predator: str, #is this predatory message label or the author is predator?
        #predatory_conversation: str, should go to conversation class
    ):

        self.idx = message_line_number
        self.author_id = author_id
        self.time = time
        self.n_chars = n_chars
        self.n_words = n_words
        self.text = text
        self.tagged_predator = tagged_predator
        
    def __repr__(self):
        return (
            f"Message Line Number: {self.message_line_number},\n"
            f"Author ID: {self.author_id},\n"
            f"Conversation Time: {self.time},\n"
            f"Message Char Count: {self.message_char_count},\n"
            f"Message Word Count: {self.message_word_count},\n"
            f"Conversation Size: {self.conversation_size},\n"
            f"Number of authors: {self.number_of_authors},\n"
            f"Message Text: {self.message_text}),\n"
            f"Tagged Predator: {self.tagged_predator},\n"
            f"Predatory Conversation: {self.predatory_conversation}\n"
        )
