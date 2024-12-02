# EE451Project

## Introduction

This repository contains the serial and parallel implementation of Conway's Game of Life.

### Serial Implementation

The serial implementation is written in gameOfLife.c file. It contains the naive implementation of Conway's Game of Life. The compile commands for this file are already written in the Makefile. 

Use the following commands to compile and run the program for the serial implementation.

```
make
./serial inputs/sparse.txt
```

For the dense dataset, replace sparse.txt with dense.txt.

### Parallel Implementation

The parallel implementation is written in the gameOfLife.cu file. It contains the parallelized code testing on different block sizes. If you want to test on a different block size, you can change the BLOCK_SIZE constant in the code. If you would like to try on different generations, change the nGenerations variable. The commands with the line arguments are written in the job.sl file. You can uncomment specific lines to test the code on different datasets. For this implementation, we ran it on USC's CARC HPC.

Use the following commands to compile and run the program for the parallel implementation. The execution time will be printed in the gpujob.out file.

```
nvcc -o gameOfLife gameOfLife.cu
sbatch job.sl
```



