import os
import csv

from cmn.message import Message


class Conversation(Object):
    def __init__(self, id, messages, participants):
        self.id = id
        self.messages = messages
        self.participants = participants

    @staticmethod
    def loader(path):
        if path.endswith(".csv"): return Conversation.csv_loader(path)

    @staticmethod
    def csv_loader(filepath):
        convs = {}

        with open(filepath, mode="r", newline="", encoding="utf-8") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            # conversation = Conversation()
            for row in csv_reader:
                # In the future, if we are having data which needs to be tweaked
                # before we import, could be taken care here.
                try:
                    # Assign the conversation id to Conversation Object
                    # conversation.conversation_id = row['conv_id']

                    message = Message(
                        row["msg_line"],
                        row["author_id"],
                        row["time"],
                        row["msg_char_count"],
                        row["msg_word_count"],
                        row["conv_size"],
                        row["nauthor"],
                        row["text"],
                        row["tagged_predator"],
                        row["predatory_conv"],
                    )

                    if id not in convs: convs[id] = Conversation(row["conv_id"], None, None)
                    convs[id].add_message(message)

                except KeyError as e:
                    print(f"Import Error: {e}")

        return convs

    def __repr__(self):
        repr_string = f"Conversation ID: {self.id}\nNumber of messages: {len(self.messages)}\n"

        if not self.messages: repr_string += "No messages found for this conversation.\n"
        else:
            for message in self.messages: repr_string += f"\n{message}"

        return repr_string
