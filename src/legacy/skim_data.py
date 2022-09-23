import xml.etree.ElementTree as ET

corpus_training_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
corpus_training_predator_id_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
dummy_training = 'D:/data/train/traindummy.xml'
corpus_test_file = 'D:/data/test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'
corpus_test_predator_id_file = 'D:/data/test/pan12-sexual-predator-identification-groundtruth-problem1.txt'
predator = False
predators = []
file = open(corpus_test_predator_id_file, "r")
counter = 0

total_conversations = 0
total_predators = 0
predators_in_binary_conversations = 0
predator_only_conversations = 0
predatorns_in_non_binary_conversations = 0

for line in file:
    # counter = counter + 1
    predators.append(line.strip())
    # print (counter)

root = ET.parse(corpus_test_file).getroot()
root2 = ET.Element("conversations")

for conversation in root.findall('conversation'):
    counter = counter + 1
    authors = []
    print(counter)
    if predator:
        predator = False
        continue
    for author in conversation.findall('message/author'):
        # print(author.text)
        # if author.text not in authors:
        #     authors.append(predators)
        if author.text in predators or counter % 50 == 0:
            root2.append(conversation)
            predator = True
            break

tree = ET.ElementTree(root2)
tree.write("predators_only_test.xml")
