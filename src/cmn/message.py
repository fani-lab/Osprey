class Message:
    def __init__(
        self,
        conversation_id: str,
        message_line_number: str,
        author_id: str,
        time: str,
        message_char_count: str,
        message_word_count: str,
        conversation_size: str,
        number_of_authors: str,
        message_text: str,
        tagged_predator: str,
        predatory_conversation: str,
    ):

        self.conversation_id = conversation_id
        self.message_line_number = message_line_number
        self.author_id = author_id
        self.time = time
        self.message_char_count = message_char_count
        self.message_word_count = message_word_count
        self.conversation_size = conversation_size
        self.number_of_authors = number_of_authors
        self.message_text = message_text
        self.tagged_predator = tagged_predator
        self.predatory_conversation = predatory_conversation

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
