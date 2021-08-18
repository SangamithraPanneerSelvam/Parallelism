# Parallelism

ResNet.py ---> A serial run of ResNet model. for epochs=10, duration is 3 hours 30 minutes.

gloo_run.py ---> Single Machine multiple process, Backend=gloo, for epochs=10, duration is 1 hour 41 minutes.

mpi_run.py ---> Backend = mpi
