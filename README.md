# Parallelism

The project involves analyzing different possibilities of trainings of Neural network on distributed systems. Training and inference of deep neural networks are computationally very intensity . This 
requires very efficient implementation to bring the training time down to a reasonable level, while preserving the inference accuracy.

1) Analysed distributed training of ResNet18 CNN model on GPU and CPU Linux HPC machine clusters to reduce computational intensity and memory demands of DNN.
2) Trained the model by Data and Model parallelism techniques with PyTorch and Message passing interface with relevance of batch size and average gradients using DP & DDP PyTorch packages
