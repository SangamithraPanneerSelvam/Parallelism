import pytorch
import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
import torchbearer
import platform
from torchvision import datasets, transforms
import argparse


parser = argparse.ArgumentParser(description='Torchbearer Distributed Data Parallel MNIST')

parser.add_argument('--rank', '-r', dest='rank', help='Rank of this process')
parser.add_argument('-n', dest='world_size', default=2, help='World size')
args = parser.parse_args()

def run(rank,size):
    print("Rank ", rank)
def init_process(rank, size, fn, backend):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size )
    fn(rank, size)

if __name__ == "__main__":
     print(torch.distributed.is_mpi_available())
#     init_process(0, 0, run, backend='mpi')
#    size=int(args.world_size)
#    print(size)
#    print(type(size))
#    processes = []
#    mp.set_start_method("spawn")
#    for rank in range(size):
#        p = mp.Process(target=init_process, args=(rank, size, run))
#        p.start()
#        processes.append(p)
               

#    for p in processes:
#        p.join()

    
