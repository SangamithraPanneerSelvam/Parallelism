from Network.Network import ResNet
from Network.Network import BasicBlock
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


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

    transform_test = transforms.Compose(
        [
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

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
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
    return (trainloader, testloader)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


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
        inputs, targets = inputs.to(device), targets.to(device)
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

    #     if (batch_idx) % 20 == 0:
    #         print(
    #             "Step_Size:[{}/{}] | Loss: %.4f ".format(
    #                 batch_idx, total_step, train_loss / (batch_idx + 1)
    #             )
    #         )
    train_acc.append(100 * correct / total)
    training_loss.append(train_loss / total_step)
    print(
        f"\ntrain-loss: {np.mean(training_loss):.3f}, train-acc: {(100 * correct/total):.4f}"
    )


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

    trainloader, testloader = load()
    net = ResNet18()

    net = nn.DataParallel(net, device_ids=[1, 2, 3, 0])
    net = net.to(device)
    criterion, optimizer, scheduler = compile(net)

    # Save checkpoint.

    for epoch in range(start_epoch, start_epoch + 1):
        train(net, trainloader, optimizer, criterion, epoch)
        # test(net, testloader, criterion, epoch)
        scheduler.step()


if __name__ == "__main__":
    main()
