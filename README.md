# Parallelism

ResNet.py ---> A serial run of ResNet model. for epochs=10, duration is 3 hours 30 minutes.
execution code ---->python ResNet.py

run.py ---> Single Machine multiple process, Backend=gloo, for epochs=10, duration is 1 hour 41 minutes onlly for 2 processes.
execution code ----> python run.py -n 2



test.py ---> Backend = mpi
