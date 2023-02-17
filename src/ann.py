import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchmetrics

from src.datasets.mydataset import MyDataset


class ANN(nn.Module):

    def __init__(self, in_features, out_features, lr, batch_size, epochs, train_dataloader, test_dataloader):
        super(ANN, self).__init__()
        self.layer_1 = nn.Linear(in_features=in_features, out_features=20)
        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.ReLU()
        self.activation_3 = nn.Softmax(dim=1)
        self.layer_2 = nn.Linear(in_features=20, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=out_features)

        self.learning_rate = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader

    def forward(self, x):
        x = self.layer_1(x)
        x = self.activation_1(x)
        x = self.layer_2(x)
        x = self.activation_2(x)
        x = self.layer_3(x)
        x = self.activation_3(x)
        return x

    def prep(self):
        pass

    def learn(self):
        # Initialize the loss function

        size = len(self.train_dataloader.dataset)
        for batch, (X, y) in enumerate(self.train_dataloader):
            # pred and loss
            pred = self.forward(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self):
        accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=2)
        precision = torchmetrics.classification.MulticlassPrecision(num_classes=2)
        recall = torchmetrics.classification.MulticlassRecall(num_classes=2)
        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = self.forward(X)

        print(f"Test Error: \n Accuracy: {(100 * accuracy(pred, y)):>0.1f}%")
        print(f"Precision : {(100 * precision(pred, y)):>0.1f}%")
        print(f"Recall    : {(100 * recall(pred, y)):>0.1f}%")

    # def eval(self):
    #     pass

    # def main(self):
    #     pass


if __name__ == '__main__':

    # dataset = pd.read_csv('./heart.csv')
    # train, test = train_test_split(dataset)
    # train_dataset = MyDataset(train[["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng",
    #                                  "oldpeak", "slp", "caa", "thall"]].values, train["output"].values)
    # test_dataset = MyDataset(test[
    #                              ["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak",
    #                               "slp", "caa", "thall"]].values, test["output"].values)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = ANN(in_features=13, out_features=2, batch_size=64, lr=0.001, epochs=10,
                train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    model.learn()
