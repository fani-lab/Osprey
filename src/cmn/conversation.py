import csv
import xml.etree.ElementTree as ET

from cmn.message import Message


class Conversation():
    def __init__(self, id, messages):
        self.id = id
        self.messages = messages

    @staticmethod
    def loader(path):
        if path.endswith(".csv"): return Conversation.csv_loader(path)
        if path.endswith(".xml"): return Conversation.xml_loader(path)

    @staticmethod
    def xml_loader(filepath):
        convs = {}
        root = ET.parse(filepath).getroot()

        for conv in root:
            conv_id = conv.attrib.get("id")
            conv_messages = []
            conv_obj = Conversation(conv_id, conv_messages)
            for message in conv:
                text = message.findtext('text')
                author = message.findtext('author')
                time = message.findtext('time')
                msg_obj = Message(author, time, text, conv_obj)
                conv_messages.append(msg_obj)
            if conv_id not in convs: 
                convs[conv_id] = conv_obj
        
        return convs
    
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