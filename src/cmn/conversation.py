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
        if path.endswith(".csv"): return Conversation.csv_loader(path)

    @staticmethod
    def csv_loader(filepath):
        convs = {}
        with open(filepath, mode="r", newline="", encoding="utf-8") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                try:
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
                    if conv_id not in convs:
                        convs[conv_id] = Conversation(conv_id)

                    convs[conv_id].messages.append(message)
                    convs[conv_id].participants.add(row["author_id"])
                    convs[conv_id].conv_size = row["conv_size"]

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
