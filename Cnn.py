import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split

import torchvision
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import torch.distributed as dist
from datetime import datetime


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

class CNN(nn.Module):
    """CNN."""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def load():
    # Data
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )


    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return (trainloader)


def compile(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return (criterion, optimizer, scheduler)

    # Training


def train(net, trainloader, optimizer, criterion, epoch):
    training_loss = []
    train_acc = []
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_step = len(trainloader)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs, targets
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print(
           batch_idx,
           len(trainloader),
           "Loss: %.3f | Acc: %.3f%% (%d/%d)"
           % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    train_acc.append(100 * correct / total)
    training_loss.append(train_loss / total_step)
    print(
        f"\ntrain-loss: {np.mean(training_loss):.3f}, train-acc: {(100 * correct/total):.4f}")


def test(net, testloader, criterion, epoch):
    val_loss = []
    val_acc = []
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_step = len(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
        val_acc.append(100 * correct / total)
        val_loss.append(test_loss / total_step)
        print(
            f"\ntest-loss: {np.mean(val_loss):.3f}, test-acc: {(100 * correct/total):.4f}"
        )


def main():
    start_time=datetime.now()
    print("TRAINING STARTS NOW : %s"% (start_time))   
    trainloader = load()
    net = CNN()
    net = net
    criterion, optimizer, scheduler = compile(net)

    # Save checkpoint.

    for epoch in range(start_epoch, start_epoch + 10):
        train(net, trainloader, optimizer, criterion, epoch)
#        test(net, testloader, criterion, epoch)
        #scheduler.step()
    end_time = datetime.now()
    print("Duration: {}".format(end_time - start_time))  


if __name__ == "__main__":
    main()
