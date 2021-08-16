import socket
from Network.Network import ResNet
from Network.Network import BasicBlock
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
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


class Net(nn.Module):
    """Network architecture."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def partition_dataset():
    """Partitioning CIFAR10"""
    dataset = datasets.CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)

    return train_set, bsz


def average_gradients(model):
    """Gradient averaging."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def compile(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return (criterion, optimizer, scheduler)


def run(rank, size):
    """Distributed Synchronous SGD Example"""
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = ResNet18()
    model = model
    #    model = model.cuda(rank)
    criterion, optimizer, scheduler = compile(model)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        train_acc = []
        training_loss = []
        epoch_loss = 0
        correct = 0
        total = 0
        total_step = len(train_set)
        for batch_idx, (data, target) in enumerate(train_set):
            data, target = Variable(data), Variable(target)
            #            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            average_gradients(model)
            epoch_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        print(
            "Rank ", dist.get_rank(), ", epoch ", epoch, ": ", epoch_loss / num_batches
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


def init_process(rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "129.69.213.175"
    os.environ["MASTER_PORT"] = "8080"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":

    start_time = datetime.now()
    print("TRAINING STARTS NOW : %s" % (start_time))
    size = 2
    #    size =dist.get_world_size()
    #    rank=dist.get_rank()
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
        print(rank)

    for p in processes:
        p.join()

    end_time = datetime.now()
    print("Duration: {}".format(end_time - start_time))
