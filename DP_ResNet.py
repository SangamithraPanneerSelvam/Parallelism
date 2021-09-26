from mpi4py import MPI
#from Network.Network import ResNet
#from Network.Network import BasicBlock

from math import ceil
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from random import Random
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime



class BasicBlock(nn.Module):
    

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
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

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
            self.in_planes=planes
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



class Partition(object):
    """Dataset-like object, but only access a subset of it."""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partitions a dataset into different chuncks."""

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(rank,size):
    """Partitioning CIFAR10"""

    # Data
    print("==> Preparing data..")
    dataset = datasets.CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
	    transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        ),
    )
    
   
    bsz = int(128 / float(size))
    partition_sizes = [1 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(rank)
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)

    return train_set, bsz

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

""" Gradient averaging. """
def average_gradients(model,size,comm):

    for param in model.parameters():
        comm.allreduce(param.grad.data,op=MPI.SUM)
        param.grad.data /= size

def compile(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return (criterion, optimizer, scheduler)

def plotting(training_loss,train_acc):
    plot1 = plt.figure(1)
    plt.plot(train_acc,'-o',color='#2ca02c')

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train'])
    plt.title('Epoch vs Accuracy')
    plt.savefig('plot1-ResNet_N=2_40_accuracy.png', dpi=300, bbox_inches='tight')
    
    plot2 = plt.figure(2)
    plt.plot(training_loss,'-o',color='#ff7f0e')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train'])
    plt.title('Epoch vs Loss')
    plt.savefig('plot2-ResNet_N=2_40_loss.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def run(rank, size,comm):
    """Distributed Synchronous SGD Example"""
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset(rank,size)
    model = ResNet18()
    model = model
    criterion, optimizer, scheduler = compile(model)
    train_acc = []
    training_loss = []
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(40):

        epoch_loss = 0
        correct = 0
        total = 0
        total_step = len(train_set)
        for batch_idx, (data, target) in enumerate(train_set):
            data, target = data,target
            #            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            average_gradients(model,size,comm)
            
            epoch_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
 

        print(
            "Processing unit: %d "% rank, ", epoch ", epoch, ": ", epoch_loss / num_batches
        )
        print(
           batch_idx,
           len(train_set),
          "Loss: %.3f | Acc: %.3f%% (%d/%d)"
           % (epoch_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )
        train_acc.append(100 * correct / total)
        training_loss.append(epoch_loss / total_step)
        print(
            f"\ntrain-loss: {np.mean(training_loss):.3f}, train-acc: {(100 * correct/total):.4f}"
        )

    plotting(training_loss, train_acc)

if __name__ == "__main__":

    start_time = datetime.now()
    print("TRAINING STARTS NOW : %s" % (start_time))
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    rank=comm.Get_rank()
    print(rank)
    run(rank,size,comm)
    end_time = datetime.now()
    print("Duration: {}".format(end_time - start_time))
