import re

import pandas as pd

from src.utils.commons import message_csv2conversation_csv, force_open, balance_dataset, create_toy_dataset, CommandObject

class CreateConversations(CommandObject):

    def get_actions_and_args(self):
        
        def create_conversations():
            df = pd.read_csv("data/dataset-v2/train.csv")
            df = message_csv2conversation_csv(df)
            with force_open("data/dataset-v2/conversation/train.csv", mode="wb") as f:
                df.to_csv(f)
                del df
            
            df = pd.read_csv("data/dataset-v2/test.csv")
            df = message_csv2conversation_csv(df)
            with force_open("data/dataset-v2/conversation/test.csv", mode="wb") as f:
                df.to_csv(f)
        return (create_conversations, [])
    
    @classmethod
    def command(cls) -> str:
        return "create-conversations"

    def help(self) -> str:
        return "creates conversations from csv file of messages"


class BalanceDatasetsForVersionTwo(CommandObject):

    def get_actions_and_args(self):

        def balance_datasets_for_version_two(ratio=0.3):
            train = "data/dataset-v2/conversation/train-v2.csv" # TODO
            test  = "data/dataset-v2/conversation/test-v2.csv"  # TODO
            
            df = pd.read_csv(train)
            train = balance_dataset(df, ratio=ratio)
            train.to_csv(f"data/dataset-v2/conversation/balanced-train-v2-{str(ratio).replace('.', '')}.csv")

            df = pd.read_csv(test)
            test = balance_dataset(df, ratio=ratio)
            test.to_csv(f"data/dataset-v2/conversation/balanced-test-v2-{str(ratio).replace('.', '')}.csv")
        
        return (balance_datasets_for_version_two, [{
                "flags": "--ratio",
                "dest": "ratio",
                "type": float,
                "default": 0.3,
                "help": "value of #predatory/(#predatory+#non-predatory)",
            }
        ])
    
    @classmethod
    def command(cls) -> str:
        return "balance-dataset-for-v2"

    def help(self) -> str:
        return "balances the distribution of labels using the original dataset to the specified ratio."


class BalanceSequentialDatasetsForVersionTwo(CommandObject):
    
    def get_actions_and_args(self):
        
        def balance_sequential_datasets_for_version_two(ratio=0.3, name_post_fix="-04"):
            train = "data/dataset-v2/train"+name_post_fix+".csv"
            test  = "data/dataset-v2/test"+name_post_fix+".csv"
            
            df = pd.read_csv(train)
            train = balance_dataset(df, ratio=ratio)
            train.to_csv(f"data/dataset-v2/train-{str(ratio).replace('.', '')}.csv")

            df = pd.read_csv(test)
            test = balance_dataset(df, ratio=ratio)
            test.to_csv(f"data/dataset-v2/test-{str(ratio).replace('.', '')}.csv")
        
        return (balance_sequential_datasets_for_version_two, [{
                "flags": "--postfix",
                "dest": "name_post_fix",
                "type": str,
                "default": "-04",
                "help": "postfix of input train and test sets. better read the code here.",
            }, {
                "flags": "--ratio",
                "dest": "ratio",
                "type": float,
                "default": 0.3,
                "help": "value of #predatory/(#predatory+#non-predatory)",
            }
        ])
    
    @classmethod
    def command(cls) -> str:
        return "balance-sequential-datasets-for-v2"

    def help(self) -> str:
        return "balances the distribution of labels in `/path/to/{--postfix}-dataset of conversation records to the specified ratio."


class CreateConversationToySet(CommandObject):

    def get_actions_and_args(self):
        
        def create_conversation_toy_set(train, test, ratio):
            df = pd.read_csv(train)
            df = create_toy_dataset(df, ratio)
            temp = re.split(r"(/|\\)", train)
            new_path = "".join(temp[:-1] + ["toy-" + temp[-1]])
            df.to_csv(new_path)
            
            temp = re.split(r"(/|\\)", test)
            new_path = "".join(temp[:-1] + ["toy-" + temp[-1]])
            df = pd.read_csv(test)
            df = create_toy_dataset(df, ratio)
            df.to_csv(new_path)
        
        return (create_conversation_toy_set, [{
                "flags": "--train-path",
                "dest": "train",
                "type": str,
                "default": "data/dataset-v2/conversation/balanced-train-v2-04.csv",
                "help": "path to train set",
            }, {
                "flags": "--test-path",
                "dest": "test",
                "type": str,
                "default": "data/dataset-v2/conversation/balanced-test-v2-04.csv",
                "help": "path to test set",
            }, {
                "flags": "--ratio",
                "dest": "ratio",
                "type": float,
                "default": 0.1,
                "help": "value of size(toy_set)/size(input_set)",
            }
        ])

    @classmethod
    def command(cls) -> str:
        return "create-toy-conversation"
    
    def help(self) -> str:
        return "create a toy set for conversations dataset"
    