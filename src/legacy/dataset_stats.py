import xml.etree.ElementTree as ET

corpus_training_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
corpus_training_predator_id_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
dummy_training = 'D:/data/train/traindummy.xml'
corpus_test_file = 'D:/data/test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'
corpus_test_predator_id_file = 'D:/data/test/pan12-sexual-predator-identification-groundtruth-problem1.txt'
predator = False
predators = []
file = open(corpus_training_predator_id_file, "r")
counter = 0
current_convo = 0

total_conversations = 0
total_predators = 0
predators_in_binary_conversations = 0
predator_only_conversations = 0
predators_in_non_binary_conversations = 0

for line in file:
    # counter = counter + 1
    predators.append(line.strip())
    # print (counter)

root = ET.parse(corpus_training_file).getroot()
# root = ET.parse('predators_only_test.xml').getroot()

root2 = ET.Element("conversations")
num_messages = 0
for conversation in root.findall('conversation'):
    counter = counter + 1
    messages = ET.Element('total_messages')
    messages.text = str(len(conversation.findall('message')))
    for message in conversation.findall('message'):
        message.append(messages)
        for text in message.findall('text'):
            print(text.text)
            message_length = ET.Element('message-length')
            if text.text is not None:
                message_length.text = str(len((text.text)))
            else:
                message_length.text = str(0)
            message.append(message_length)
    # print(len(conversation.findall('message')))
    # print(counter)
    # if predator:
    #     predator = False
    #     continue
    # for author in conversation.findall('message/author'):
    #     authors.append(author.text)
    # is_pred, number_of_predators = common_member(authors, predators)
    # if is_pred:
    #     if number_of_predators == 1:
    #         predators_in_binary_conversations += 1
    #     if number_of_predators > 1:
    #         predators_in_non_binary_conversations += 1
    #
    # authors = []

# print('Average Messages per conversation: ', num_messages / counter)
# print("predators_in_binary_conversations", predators_in_binary_conversations)
# print("predators_in_non_binary_conversations", predators_in_non_binary_conversations)
tree = ET.ElementTree(root)
tree.write("messagestats.xml")