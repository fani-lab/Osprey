import csv

from cmn.message import Message


class Conversation:
    def __init__(self, id):
        self.id = id
        self.messages = []
        self.participants = set()
        self.conv_size = 0

    @staticmethod
    def loader(path):
        if path.endswith(".csv"):
            return Conversation.csv_loader(path)

    def add_message(self, message_content, author_involved, size):
        self.messages.append(message_content)
        self.participants.add(author_involved)
        self.conv_size = size

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

                    message = Message(
                        row["msg_line"],
                        row["author_id"],
                        row["time"],
                        row["msg_char_count"],
                        row["msg_word_count"],
                        row["text"],
                        row["tagged_predator"],
                    )

                    conv_id = row["conv_id"]
                    author_involved = row["author_id"]
                    conversation_size = row["conv_size"]

                    if conv_id not in convs:
                        convs[conv_id] = Conversation(conv_id)

                    convs[conv_id].add_message(
                        message, author_involved, conversation_size
                    )

                except KeyError as e:
                    print(f"Import Error: {e}")

        return convs

    def __repr__(self):
        authors_list = "\n".join(self.participants)

        repr_string = f"Conversation ID: {self.id}\nConversation Size: {self.conv_size}\nAuthors Involved: {len(list(self.participants))}\n{authors_list}\n"

        if not self.messages:
            repr_string += "No messages found for this conversation.\n"
        else:
            for message in self.messages:
                repr_string += f"\n{message}"

        return repr_string
