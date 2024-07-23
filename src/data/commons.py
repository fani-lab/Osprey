from typing import List, Optional, Union, Dict
import logging
import functools

logger = logging.getLogger()


functools.total_ordering
class Message:
    def __init__(self) -> None:
        self.text = ""
        self.conversation_id = None
        self.message_id = None
        self.message_line = 0
        self.tagged_predator = False
        self.message_timestamp = 0
        self.author_id = None
        self.backtranslation_text = ""
        self.tokens = None
        # the following are appended when the dataset retrieves them
        self.text_vectors = None
        self.nontext_vectors = None
        self.aggregated_conversation = False # if true, message_id is None and all conversation is represented as single a message
        # todo: you can define other statistical fields for each message

    
    # assuming both have the same conversation_id 
    def __lt__(self, other) -> bool:
        if self.conversation_id != other.conversation_id:
            raise ValueError("comparing messages from two different conversations")
        return self.message_line < other.message_line
    
    def __eq__(self, other) -> bool:
        if self.conversation_id != other.conversation_id:
            raise ValueError("comparing messages from two different conversations")
        return self.message_line == other.message_line
    

class Conversation:

    def __init__(self) -> None:
        self.conversation_id = None
        self.messages = [] # list of Message objects
        self.is_predatory_conversation = False # if one predator is present
        self.is_aggregated_conversation = False # works the same as in message
        self.authors_ids = set()
        self.messages_count = 0
        self.langauge = 'en' # default
        self.backtranslation_language = None
        self.backtranslation_translator = None
    
    def get_processed_conversation_id(self) -> str:
        if self.backtranslation_language is None:
            return self.conversation_id
        return self.conversation_id + "_" + self.backtranslation_translator + "_" + self.backtranslation_language

    def __eq__(self, value) -> bool:
        return self.get_processed_conversation_id() == value.get_processed_conversation_id()

