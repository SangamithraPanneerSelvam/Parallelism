
"""Import libraries"""
# exection statement : mpirun -n python script.py
# mpi4py python packages is used to get the rank and size.
# Data Parallelism in CPU and GPU has slight difference between each other.  
# Hence remove the # mentioned inside the codes to implement them either in cpu or gpu.


from mpi4py import MPI
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from math import ceil
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import matplotlib
from random import Random
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime
from torch.autograd import Variable

""" ResNet18 build with basic block and resnet block """

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
        out = F.softmax(out)
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

""" DataPartition and Partition class for manual data split and distribute """

class DataPartition(object):
    """Partitions a dataset into different chuncks."""

    def __init__(self, dataset, partiton_sizes=[], seed=1234):
        self.dataset = dataset
        self.partition_list = []

    def __shuffle__(self):
        random_no_gen = Random()
        random_no_gen.seed(seed)
        dataset_len = len(self.dataset)
        ids = [x for x in range(0, dataset_len)]
        random_no_gen.shuffle(ids)

        for i in partition_sizes:
            partition_len = int(i * dataset_len)
            self.partition_list.append(ids[0:partition_len])
            ids = ids[partition_len:]

    def part(self, partition):
        return Partition(self.dataset, self.partition_list[partition])

class Partition(object):
    """Dataset-like object, but only access a subset of it."""

    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = index

    def __getlength__(self):
       return len(self.index)

    def __getid__(self, index):
        dataset_idx = self.index[index]
        return self.dataset[dataset_idx]

""" Loading the dataset and partitioning dataset"""

def partition_dataset(rank,size):
    """Partitioning CIFAR10"""
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
    """DataParallelism in GPU"""
#    batch_size = 128
    """DataParallelism in CPU"""
    batch_size = int(128 / float(size))
    
    partition_sizes = [1 / size for _ in range(size)]
    obj_part = DataPartitioner(dataset, partition_sizes)
    obj_part = obj_part.part(rank)
    train_set = torch.utils.data.DataLoader(obj_part, batch_size=batch_size, shuffle=True)

    return train_set, batch_size


""" Returns the ResNet18 network"""
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


""" Gradient averaging on CPU  """
def average_gradients(model,size,comm):

    for param in model.parameters():
       comm.allreduce(param.grad.data, op=MPI.SUM)
       param.grad.data/=size


""" Compile network """
def compile(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    return (criterion, optimizer)


""" To plot the  graphs : Accuracy, loss"""
def plotting(loss,accuracy):

    plot1 = plt.figure(1)
    plt.plot(accuracy,'-o')

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train'])
    plt.title('Epoch vs Accuracy')
    plt.savefig('plot1-accuracy.png', dpi=300, bbox_inches='tight')

    plot2 = plt.figure(2)
    plt.plot(loss,'-o')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train'])
    plt.title('Epoch vs Loss')
    plt.savefig('plot2-loss.png', dpi=300, bbox_inches= 'tight')


""" Training of Neural network """
def run(rank, size,comm):
  
    torch.manual_seed(1234)
    train_set, batch_size = partition_dataset(rank,size)
    model = ResNet18()

    """dataparallelism in cpu"""
    model= model

    """dataparallelism in GPU"""

#    model = nn.DataParallel(model,device_ids=[rank])
#    model= model.cuda(rank)
    
    criterion, optimizer = compile(model)
    model.train()
    accuracy = []
    loss = []
    train_time=[]
    num_batches = ceil(len(train_set.dataset) / float(batch_size))
    
    for epoch in range(40):
        epoch_loss = 0
        correct = 0
        total = 0
        total_step = len(train_set)
        for batch_idx, (data, target) in enumerate(train_set):

            """ dataparallelism in cpu :"""
            data, target = data,target

            """dataparallelism in GPU"""
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            """DataParallel in CPU"""
            average_gradients(model,size,comm)

            epoch_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()


        print("Processing unit: %d "% rank, ", epoch ", epoch, ": ", epoch_loss / num_batches)
   
        accuracy.append(100 * correct / total)
        loss.append(epoch_loss / total_step)
        print(
            f"\ntrain-loss: {np.mean(loss):.3f}, train-acc: {(100 * correct/total):.4f}"
        )

    plotting(loss, accuracy)

""" main function"""
def main():

    start_time = datetime.now()
    print("TRAINING STARTS NOW : %s" % (start_time))
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    rank=comm.Get_rank()
    print(rank)
    run(rank,size,comm)
    end_time = datetime.now()
    print("Duration: {}".format(end_time - start_time))


if __name__ == "__main__":
    main()