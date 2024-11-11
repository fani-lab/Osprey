import os
import csv

from cmn.message import Message


class Conversation(Message):
    def __init__(self, conversation_id):
        self.messages = []
        self.conversation_id = conversation_id

    @staticmethod
    def loader(path):
        if path.endswith(".csv"):
            return Conversation.csv_loader(path)

    def add_message(self, message_content):
        self.messages.append(message_content)

    @staticmethod
    def csv_loader(filepath):
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' not found.")
            return

        conversation_dict = {}

        with open(filepath, mode="r", newline="", encoding="utf-8") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            # conversation = Conversation()
            for row in csv_reader:
                # In the future, if we are having data which needs to be tweaked
                # before we import, could be taken care here.
                try:
                    # Assign the conversation id to Conversation Object
                    # conversation.conversation_id = row['conv_id']

                    # import the data from messages
                    conversation_id = row["conv_id"]
                    message_line_number = row["msg_line"]
                    author_id = row["author_id"]
                    time = row["time"]
                    message_char_count = row["msg_char_count"]
                    message_word_count = row["msg_word_count"]
                    conversation_size = row["conv_size"]
                    number_of_authors = row["nauthor"]
                    message_text = row["text"]
                    tagged_predator = row["tagged_predator"]
                    predatory_conversation = row["predatory_conv"]

                    message = Message(
                        conversation_id,
                        message_line_number,
                        author_id,
                        time,
                        message_char_count,
                        message_word_count,
                        conversation_size,
                        number_of_authors,
                        message_text,
                        tagged_predator,
                        predatory_conversation,
                    )

                    if conversation_id not in conversation_dict:
                        conversation_dict[conversation_id] = Conversation(
                            conversation_id
                        )

                    conversation_dict[conversation_id].add_message(message)

                except KeyError as e:
                    print(f"Import Error: {e}")

        return conversation_dict

    def __repr__(self):
        conv_messages = [
            message
            for message in self.messages
            if message.conversation_id == self.conversation_id
        ]
        length_of_messages = len(conv_messages)

        repr_string = f"Conversation ID: {self.conversation_id}\nNumber of messages: {length_of_messages}\n"

        if not conv_messages:
            repr_string += "No messages found for this conversation.\n"
        else:
            for message in conv_messages:
                repr_string += f"\n{message}"

        return repr_string
