import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, train, target):
        self.data = torch.from_numpy(train).to(torch.float32)
        self.labels = torch.from_numpy(target)

    def __sizeof__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

