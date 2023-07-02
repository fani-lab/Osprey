import re

import pandas as pd

from src.utils.commons import message_csv2conversation_csv, force_open, balance_dataset, create_toy_dataset, pan12_xml2csv, CommandObject


class XML2CSV(CommandObject):
    
    def get_actions_and_args(self):

        def callback(xmlfile, predatorsfile, output):
            df = pan12_xml2csv(xmlfile, predatorsfile)
            df.to_csv(output, sep=",")

        return callback, [{
                "flags": "--xml-file",
                "dest": "xmlfile",
                "type": str,
                "help": "path to xml file of conversations",
            },
            {
                "flags": "--predators-file",
                "dest": "predatorsfile",
                "type": str,
                "help": "path to file of predators id",
            },
            {
                "flags": "--output-file",
                "dest": "output",
                "type": str,
                "help": "path where the generated csv will be saved",
            },
        ]
    
    @classmethod
    def command(cls) -> str:
        return "xml2csv"
    
    def help(self) -> str:
        return "turns the pan12 xml file to csv which can be used by other scripts and commands."


class CreateConversations(CommandObject):

    def get_actions_and_args(self):
        
        def create_conversations(datasets_path, output_path):
            df = pd.read_csv(f"{datasets_path}train.csv")
            df = message_csv2conversation_csv(df)
            with force_open(f"{output_path}train.csv", mode="wb") as f:
                df.to_csv(f)
                del df
            
            df = pd.read_csv(f"{datasets_path}test.csv")
            df = message_csv2conversation_csv(df)
            with force_open(f"{output_path}test.csv", mode="wb") as f:
                df.to_csv(f)
        
        return (create_conversations, [{
                "flags": "--datasets-path",
                "dest": "datasets_path",
                "type": str,
                "default": "data/dataset-v2/",
                "help": "path to message base records where there is a train.csv and test.csv file",
            }, {
                "flags": "--output-path",
                "dest": "output_path",
                "type": str,
                "default": "data/dataset-v2/conversation/",
                "help": "path to directory where the resulting conversation dataframe will be saved as CSV",
            },
        ])
    
    @classmethod
    def command(cls) -> str:
        return "create-conversations"

    def help(self) -> str:
        return "creates conversations from csv file of messages"


class BalanceDatasetsForVersionTwo(CommandObject):

    def get_actions_and_args(self):

        def balance_datasets_for_version_two(datasets_path, output_path, ratio=0.3):
            train = f"{datasets_path}train-v2.csv" # TODO
            test  = f"{datasets_path}test-v2.csv"  # TODO
            
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
            }, {
                "flags": "--datasets-path",
                "dest": "datasets_path",
                "type": str,
                "default": "data/dataset-v2/conversation/",
                "help": "path to message base records where there is a train-v2.csv and test-v2.csv file",
            }, {
                "flags": "--output-path",
                "dest": "output_path",
                "type": str,
                "default": "data/dataset-v2/conversation/",
                "help": "path to directory where the resulting conversation dataframe will be saved as CSV with the name 'balanced-{test/test}-v2-{ratio}.csv'",
            },
        ])
    
    @classmethod
    def command(cls) -> str:
        return "balance-dataset-for-v2"

    def help(self) -> str:
        return "balances the distribution of labels using the original dataset to the specified ratio."


class BalanceSequentialDatasetsForVersionTwo(CommandObject):
    
    def get_actions_and_args(self):
        
        def balance_sequential_datasets_for_version_two(trainset, testset, output_path, ratio=0.3):

            df = pd.read_csv(trainset)
            train = balance_dataset(df, ratio=ratio)
            train.to_csv(f"{output_path}train-{str(ratio).replace('.', '')}.csv")

            df = pd.read_csv(testset)
            test = balance_dataset(df, ratio=ratio)
            test.to_csv(f"{output_path}test-{str(ratio).replace('.', '')}.csv")
        
        return (balance_sequential_datasets_for_version_two, [{
                "flags": "--ratio",
                "dest": "ratio",
                "type": float,
                "default": 0.3,
                "help": "value of #predatory/(#predatory+#non-predatory)",
            }, {
                "flags": "--trainset-path",
                "dest": "trainset",
                "type": str,
                "default": "data/dataset-v2/train-02.csv",
                "help": "path to train dataset",
            }, {
                "flags": "--testset-path",
                "dest": "testset",
                "type": str,
                "default": "data/dataset-v2/test-02.csv",
                "help": "path to test dataset",
            }, {
                "flags": "--output-path",
                "dest": "output_path",
                "type": str,
                "default": "data/dataset-v2/",
                "help": "path to directory where the resulting dataframe will be saved as CSV with the name '{test/test}-{ratio}.csv'",
            },
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
        return "create a toy set for conversations dataset. The output path will be the same as input just a 'toy-' prefix will be added to the file name at the end."
    