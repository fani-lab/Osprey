## Data stats

--------------------------------
**PAN** is a series of scientific events and shared tasks on digital text forensics and stylometry[^1]

https://pan.webis.de/clef12/pan12-web/sexual-predator-identification.html
https://pan.webis.de/downloads/publications/papers/inches_2012.pdf

Public link to download: Not available

**Labels for Test Set:**
- ids of predators (one per line).
pan12-sexual-predator-identification-groundtruth-problem1.txt

- suspicious (of a perverted behavior) messages: ids of conversations and message line# in the conversation
pan12-sexual-predator-identification-groundtruth-problem2.txt 

**Labels for Train Set:**
- ids of predators (one per line).
pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt

### Columns

| Column          | Description                         |
| --------------- | ----------------------------------- |
| conv_id         | Id of conversation                  |
| msg_line        | number of the message               |
| author_id       | the ID of author                    |
| time            | the time that message sent          |
| msg_char_count  | number of characters in the message |
| msg_word_count  | number of words in the message      |
| conv_size       | length of the conversation          |
| nauthor         | number of authors in the chat       |
| text            | content of the message              |
| tagged_msg      | label of the predatory message      |
| tagged_conv     | label of predatory conversation     |
| tagged_predator | label of predator                   |

### stats

| Stat	                                               | Train  | Test    |
|-----------------------------------------------------|--------|---------|
| Total Conversations                                 | 66927  | 155128  |
| Messages (rows)                                     | 903607 | 2058781 |
| Avg Messages in Conversation                        | 13.5   | 13.27   |
| unary conversations                                 | 12773  | 29561   |
| Binary Conversations                                | 45741  | 105862  |
| N-ary Conversations                                 | 8413   | 19705   |
| Users                                               | 97689  | 218702  |
| Predatory Conversations                             | 2016   | 3737    |
| Predators                                           | 142      | 254       |
| Binary Predatory Conversations                      | 2016   | 3737    |
| Non-binary Perdatory Conversations                  | 0      | 0       |
| Avg Messages in Predatory Conversation              | 60.73  | 90.07   |
|average message number for a non-predatory conversation| 12.74  | 12.86   |
| more than 1 Predators in a Conversation             | 0      | 0       |
| Avg Ratio of Conversation (#message) for a Predator | 60.73  | 90.07   |
| Predators in Normal Conversations                   | 1997      | 2903       |

[^1]: https://pan.webis.de/
