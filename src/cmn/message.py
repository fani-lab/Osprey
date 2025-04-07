class Message():
    def __init__(
        self,
        idx: str,
        author_id: str,
        time: str,
        n_chars: str,
        n_words: str,
        text: str,
        tagged_predator: str,  # is this predatory message label or the author is predator?
    ):

        self.idx = idx
        self.author_id = author_id
        self.time = time
        self.n_chars = n_chars
        self.n_words = n_words
        self.text = text
        self.tagged_predator = tagged_predator

    def __repr__(self):
        return (
            f"Message Line Number: {self.idx},\n"
            f"Author ID: {self.author_id},\n"
            f"Conversation Time: {self.time},\n"
            f"Message Char Count: {self.n_chars},\n"
            f"Message Word Count: {self.n_words},\n"
            f"Message Text: {self.text}),\n"
            f"Tagged Predator: {self.tagged_predator},\n"
        )
