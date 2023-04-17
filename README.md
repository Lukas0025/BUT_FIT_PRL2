# Implementation of paraller k-means clustering in MPI
### PRL Project 2
### Author: Lukas Plevac <xpleva07@vutbr.cz>

This repository implementing paraller k-means clustering algoritm in cpp and MPI.

This Repository is exmaple how to use MPI, but is not emxample what do with MPI, because this algoritm is in real slower that seqvetion version, because comanication is match expensive. To make this program eficient numbers processing on ranks must be mutch harder, sort numbers to arrays is too match simple.

## Building program

```sh
mpic++ --prefix /usr/local/share/OpenMPI -o parkmeans parkmeans.cc
```

## Running program

Simpli run program with mpi run. Program need imput file with name **numbers** in same dir. Number of proccess must by devider of number of numbers (eg. for 10 numbers can be 1,2,5,10 proccesses).

```sh
dd if=/dev/random bs=1 count=10 of=numbers # create imput file with 10 numbers
mpirun --prefix /usr/local/share/OpenMPI --oversubscribe -np 5 parkmeans # run program on 5 processes
```

## Run using run.sh

run.sh contains basic command to generate file build program and run program with file.

```sh
bash run.sh #with 10 numbers
bash run.sh 64 #with 64 numbers
```
