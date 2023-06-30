import pandas as pd

from src.utils.commons import get_stats_v2, CommandObject


def _flatten_stat_dict(stats):
    stats_names = stats.keys()
    flat_dict = dict()
    for k in list(stats_names):
        item = stats[k]
        name = k.replace("_", " ")
        if type(item) == dict:
            for sub_key, sub_item in item.items():
                flat_dict[name + " & " + sub_key] = sub_item
        else:
            flat_dict[name] = item

    return flat_dict


class GenerateStats(CommandObject):

    def get_actions_and_args(self):
        
        def action():
            train = pd.read_csv("data/dataset-v2/train.csv")
            train_stats = get_stats_v2(train)
            train_stats = _flatten_stat_dict(train_stats)

            test  = pd.read_csv("data/dataset-v2/test.csv")
            test_stats = get_stats_v2(test)
            test_stats = _flatten_stat_dict(test_stats)
            result = ["|Stat	| Train | Test|Test ∪ Train\n","|-----|------|------|------|\n"]
            keys = list(train_stats.keys())
            keys.sort()
            for k in keys:
                result.append(f"|{k}|{(train_stats[k]):>0.3f}|{(test_stats[k]):>0.3f}|{(train_stats[k]+test_stats[k]):>0.3f}|\n")
            with open("stats_as_readme_table.txt", mode="w+") as f:
                f.writelines(result)
        
        return (action, dict())

    @classmethod
    def command(cls) -> str:
        return "generate-stats"

    def help(self) -> str:
        return "generates stats of the dataset"
    