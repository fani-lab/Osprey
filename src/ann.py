import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset



class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.learning_rate = None
        self.batch_size = None
        self.epochs = None
        self.optimizer = None
        self.loss_fn = None
        self.activation = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None

    def init(self, in_features, out_features, lr, batch_size, epochs):
        self.layer_1 = nn.Linear(in_features=in_features, out_features=50)
        self.activation = nn.ReLU()
        self.layer_2 = nn.Linear(in_features=50, out_features=30)
        self.layer_3 = nn.Linear(in_features=30, out_features=out_features)

        self.learning_rate = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.activation(x)
        x = self.layer_2(x)
        x = self.activation(x)
        x = self.layer_3(x)
        return x

    def prep(self):
        pass

    def learn(self, learn_dataloader):
        # Initialize the loss function

        size = len(learn_dataloader.dataset)
        for batch, (X, y) in enumerate(learn_dataloader):
            # pred and loss
            pred = self.forward(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                pred = self.forward(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += 1 if (pred.argmax(1) == y) else 0

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class MyDataset(Dataset):
    def __init__(self, df):
        self.data = torch.from_numpy(df.drop('output', axis=1).to_numpy()).to(torch.float32)
        self.labels = torch.from_numpy(df['output'].to_numpy())

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    dataset = pd.read_csv('FILL HERE')
    train, test = train_test_split(dataset)
    train_dataset = MyDataset(train)
    test_dataset = MyDataset(test)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.dtype}")
    print(90 * '~')
    print(f"Labels batch shape: {train_labels.dtype}")

    model = ANN()
    model.init(in_features=13, out_features=2, batch_size=64, lr=1e-7, epochs=10)

    for t in range(model.epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # learn
        model.learn(learn_dataloader=train_dataloader)
        # test
        model.test(test_dataloader)
    print("Done!")