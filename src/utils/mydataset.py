import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, train, target):
        self.data = torch.stack(train)
        self.labels = torch.from_numpy(target)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
