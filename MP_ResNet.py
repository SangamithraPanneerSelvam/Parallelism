"""Import libraries"""
# exection statement : mpirun -n python script.py
# Model Parallelism in CPU and GPU has slight difference between each other.  
# Hence use .cuda() for GPU and .cpu() for CPU. 
#Note: .cuda() allows you to pass the number of GPUs and .cpu() doesn't take arguments.

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
import matplotlib.pyplot as plt



""" ResNet18 build with basic block and resnet block """
class BasicBlock(nn.Module):
   expansion=1

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

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).cuda(2)
        self.bn1 = nn.BatchNorm2d(64).cuda(2)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1).cuda(2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2).cuda(3)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2).cuda(4)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2).cuda(5)
        self.linear = nn.Linear(512 * block.expansion, num_classes).cuda(5)

    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out =F.relu(self.bn1(self.conv1(x.cuda(2))))
        out = self.layer1(out.cuda(2))
        out = self.layer2(out.cuda(3))
        out = self.layer3(out.cuda(4))
        out = self.layer4(out.cuda(5))
        out = F.avg_pool2d(out, 4).cuda(5)
        out = out.view(out.size(0), -1)
        out = self.linear(out.cuda(5))
        return out

""" Loading the dataset"""
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
    train_set = torch.utils.data.DataLoader(
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
    return train_set


""" Returns the ResNet18 network"""
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


""" Compile network """
def compile(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
  
    return (criterion, optimizer)

""" To plot the  graphs : Accuracy, loss"""
def plotting(loss, accuracy):
    plot1 = plt.figure(1)
    plt.plot(accuracy, "-o")

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["Train"])
    plt.title("Epoch vs Accuracy")
    plt.savefig("plot1-accuracy.png", dpi=300, bbox_inches="tight")

    plot2 = plt.figure(2)
    plt.plot(loss, "-o")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Train"])
    plt.title("Epoch vs Loss")
    plt.savefig("plot2-loss.png", dpi=300, bbox_inches="tight")

    plt.show()


""" Training of Neural network """
def run(model, train_set, epoch, loss, accuracy):

    print("\nEpoch: %d" % epoch)
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    total_step = len(train_set)
    criterion, optimizer = compile(model)

    for batch_idx, (data, target) in enumerate(train_set):
        data, target = data.cuda(2), target.cuda(5)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    

    accuracy.append(100 * correct / total)
    loss.append(epoch_loss / total_step)
    print(
        f"\ntrain-loss: {np.mean(loss):.3f}, train-acc: {(100 * correct/total):.4f}"
    )


""" main function"""
def main():
    start_time = datetime.now()
    print("TRAINING STARTS NOW : %s" % (start_time))
    train_set = load()
    model = ResNet18()
    model = model
    loss = []
    accuracy = [] 

    for epoch in range(0, 40):
        run(model, train_set, epoch, loss, accuracy)
    
    plotting(loss, accuracy)
    end_time = datetime.now()
    print("Duration: {}".format(end_time - start_time))


if __name__ == "__main__":
    main()
