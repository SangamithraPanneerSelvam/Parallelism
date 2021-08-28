# from Network.Network import ResNet
# from Network.Network import BasicBlock
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch.distributed as dist
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            #             self.in_planes = planes * block.expansion
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def load():
    # Data
    print("==> Preparing data..")
    dataset = datasets.CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        ),
    )
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=0
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
    return trainloader


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
    start_time = datetime.now()
    print("TRAINING STARTS NOW : %s" % (start_time))
    trainloader = load()
    net = ResNet18()
    net = net
    criterion, optimizer, scheduler = compile(net)

    # Save checkpoint.

    for epoch in range(start_epoch, start_epoch + 10):
        train(net, trainloader, optimizer, criterion, epoch)
        # test(net, testloader, criterion, epoch)
        scheduler.step()
    end_time = datetime.now()
    print("Duration: {}".format(end_time - start_time))


if __name__ == "__main__":
    main()
