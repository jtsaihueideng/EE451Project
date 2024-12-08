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

The parallel implementation optimized for dense datasets is written in the gameOfLife.cu file. It contains the parallelized code testing on different block sizes. If you want to test on a different block size, you can change the BLOCK_SIZE constant in the code. If you would like to try on different generations, you can either change the nGenerations variable or add the generations value as a command line argument after the dataset file name. The commands with the line arguments are written in the job.sl file. You can uncomment specific lines to test the code on different datasets. For this implementation, we ran it on USC's CARC HPC.

Use the following commands to compile and run the program for the parallel implementation. The execution time will be printed in the gpujob.out file.

```
nvcc -o gameOfLife gameOfLife.cu
sbatch job.sl
```

The parallel implementation optimized for sparse datasets is provided in sparseGameOfLifeV2.cu. This version includes a conditional statement to check whether a neighbor is alive before incrementing the neighbor count. To compile and run this version, use the same command as before, but replace the file name with sparseGameOfLifeV2.cu.

Additionally, we implemented an optimization using a sparse representation that tracks only live cells. However, the speedup achieved with this approach was limited due to the additional overhead required to maintain the list of live cells. It can similarly be ran with the same command, but with the file name replaced with sparseGameOfLife.cu.

