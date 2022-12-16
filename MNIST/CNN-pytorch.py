import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm


BATCH_SIZE = 64

def load_data():
    train_dataset = DataLoader(MNIST("/files/", train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ])), 
                        batch_size=BATCH_SIZE, 
                        shuffle=True)
    test_dataset = DataLoader(MNIST("/files/", train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ])), 
                        batch_size=BATCH_SIZE*16, 
                        shuffle=True)

    return train_dataset, test_dataset


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(  ## (x0, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential( ## (x0, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential( ## (x0, 64, 7, 7)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1)
        )

        self.lin1 = nn.Sequential( ## (x0, 128, 4, 4)
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU()
        )
        self.lin2 = nn.Linear(512, 10)
        
        self.layers = [self.conv1, self.conv1, self.conv1, self.lin1, self.lin2]
        self.params = [p for layer in self.layers for p in layer.parameters()]

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.lin1(out.view(out.shape[0], -1))
        return self.lin2(out)

    def train(self, X, y, lr=0.1):
        for param in self.params:
            param.require_grad = True

        logits = self.forward(X)
        loss = F.cross_entropy(logits, y)

        for param in self.params:
            param.grad = None

        loss.backward()
        for param in self.params:
            param.data += -lr * param.grad

        return loss.item()


def train(model, dataset, epochs=1):
    print("--- Model Training ---")
    for epidx in range(1, epochs+1):
        pbar = tqdm(dataset)
        for _, (data, target) in enumerate(pbar):
            loss = model.train(data, target, lr=1/(10**epidx))
            pbar.set_description(f"Epoch {epidx} | Loss {loss:.4f}")


def test(model, dataset):
    total_correct = 0
    total_all = 0
    for bidx, (data, target) in enumerate(dataset):
        out = model.forward(data)
        correct = (out.argmax(axis=1) == target).sum()
        total_correct += correct
        total_all += out.shape[0]

        print(f"Idx: {bidx} | Batch Acc: {100*correct/out.shape[0]:.2f}% | Total Acc: {100*total_correct/total_all:.2f}%")
    
    print(f"Final Accuracy: {100*total_correct/total_all:.2f}%")


def main():
    model = Model()
    train_dataset, test_dataset = load_data()

    train(model, train_dataset, epochs=2)
    test(model, test_dataset)


if __name__ == "__main__":
    main()
