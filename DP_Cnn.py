from mpi4py import MPI

#from Network.Network import ResNet
#from Network.Network import BasicBlock
import torch.distributed as dist
from math import ceil
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models
from random import Random
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime

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


def partition_dataset(rank,size):
    """Partitioning CIFAR10"""

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
   
    bsz = int(128 / float(size))

    partition_sizes = [1 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(rank)
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)

    return train_set, bsz


def compile(model,size):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return (criterion, optimizer, scheduler)


def average_gradients(model,comm,size):

    for param in model.parameters():
 
      comm.allreduce(param.grad.data , op=MPI.SUM)  
      param.grad.data /= size

def run(rank, size,comm):
    """Distributed Synchronous SGD Example"""
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset(rank,size)
    model = CNN()
    model = model
    #    model = model.cuda(rank)
    criterion, optimizer, scheduler = compile(model,size)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(25):
        train_acc = []
        training_loss = []
        epoch_loss = 0
        correct = 0
        total = 0
        total_step = len(train_set)
        for batch_idx, (data, target) in enumerate(train_set):
            data, target = data, target
            #            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            average_gradients(model,comm,size)
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

