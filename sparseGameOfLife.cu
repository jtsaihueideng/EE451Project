#include <iostream>
#include <cuda.h>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MAX_LIVE_CELLS 1000000  // Adjust as needed for maximum expected live cells
#define BLOCK_SIZE 256          // Define BLOCK_SIZE for CUDA threads per block

typedef struct {
    int row;
    int col;
} Cell;

__global__ void updateNeighborCounts(Cell *liveCells, int numLiveCells, int *neighborCounts, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numLiveCells) {
        int row = liveCells[idx].row;
        int col = liveCells[idx].col;

        // Offsets to reach the 8 neighbors
        int neighborOffsets[8][2] = {{-1, -1}, {-1,  0}, {-1,  1},
            { 0, -1}, { 0,  1}, { 1, -1}, { 1,  0}, { 1,  1}};

        for (int i = 0; i < 8; ++i) {
            int nRow = row + neighborOffsets[i][0];
            int nCol = col + neighborOffsets[i][1];

            // Skip out-of-bounds neighbors
            if (nRow >= 0 && nRow < n && nCol >= 0 && nCol < n) {
                int index = nRow * n + nCol;

                // Atomically increment the neighbor count
                atomicAdd(&neighborCounts[index], 1);
            }
        }
    }
}

__global__ void applyGameOfLifeRules(Cell *liveCells, int numLiveCells, int *neighborCounts, int n, Cell *newLiveCells, int *newNumLiveCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process live cells and their neighbors
    int totalCells = n * n;
    for (int i = idx; i < totalCells; i += blockDim.x * gridDim.x) {
        int row = i / n;
        int col = i % n;
        int liveNeighbors = neighborCounts[i];

        // Check if the cell is currently alive
        bool isAlive = false;
        for (int j = 0; j < numLiveCells; ++j) {
            if (liveCells[j].row == row && liveCells[j].col == col) {
                isAlive = true;
                break;
            }
        }

        bool nextAlive = false;
        if (isAlive) {
            if (liveNeighbors == 2 || liveNeighbors == 3)
                nextAlive = true;
        } else {
            if (liveNeighbors == 3)
                nextAlive = true;
        }

        if (nextAlive) {
            int pos = atomicAdd(newNumLiveCells, 1);
            newLiveCells[pos].row = row;
            newLiveCells[pos].col = col;
        }

        // Reset neighbor count for next generation
        neighborCounts[i] = 0;
    }
}

void printBoard(const Cell* liveCells, int numLiveCells, int n, int startRow = 0, int startCol = 0, int rows = 10, int cols = 10) {
    int *board = (int*)calloc(n * n, sizeof(int));
    for (int i = 0; i < numLiveCells; ++i) {
        int idx = liveCells[i].row * n + liveCells[i].col;
        board[idx] = 1;
    }
    for (int i = startRow; i < startRow + rows && i < n; ++i) {
        for (int j = startCol; j < startCol + cols && j < n; ++j) {
            std::cout << board[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    free(board);
}

void parseMatrix(const char *filename, Cell **liveCells, int *numLiveCells, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Read the file into a buffer
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = (char *)malloc(length + 1);
    if (!buffer) {
        perror("Error allocating memory");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fread(buffer, 1, length, file);
    buffer[length] = '\0';
    fclose(file);

    // Variables to track dimensions
    *rows = 0;
    *cols = 0;

    // First pass: Determine the matrix dimensions
    int firstRow = 1;
    for (int i = 0; buffer[i] != '\0'; ++i) {
        if (buffer[i] == '[' && buffer[i + 1] != '[') {
            (*rows)++;
            if (firstRow) {
                // Count the number of columns in the first row
                for (int j = i + 1; buffer[j] != ']'; ++j) {
                    if (buffer[j] == ',') {
                        (*cols)++;
                    }
                }
                (*cols)++; // Account for the last column
                firstRow = 0;
            }
        }
    }

    printf("Parsing Matrix %d x %d\n", *rows, *cols);

    // Second pass: Parse and store live cells
    int index = 0;
    int row = 0;
    int col = 0;
    *numLiveCells = 0;
    for (int i = 0; buffer[i] != '\0'; ++i) {
        if (buffer[i] == '[' && buffer[i + 1] != '[') {
            col = 0;
            i++;
            while (buffer[i] != ']') {
                if (isdigit(buffer[i])) {
                    int val = buffer[i] - '0';
                    if (val == 1) {
                        (*numLiveCells)++;
                    }
                    col++;
                }
                i++;
            }
            row++;
        }
    }

    // Allocate memory for live cells
    *liveCells = (Cell*)malloc(*numLiveCells * sizeof(Cell));

    // Third pass: Store live cells
    index = 0;
    row = 0;
    col = 0;
    for (int i = 0; buffer[i] != '\0'; ++i) {
        if (buffer[i] == '[' && buffer[i + 1] != '[') {
            col = 0;
            i++;
            while (buffer[i] != ']') {
                if (isdigit(buffer[i])) {
                    int val = buffer[i] - '0';
                    if (val == 1) {
                        (*liveCells)[index].row = row;
                        (*liveCells)[index].col = col;
                        index++;
                    }
                    col++;
                }
                i++;
            }
            row++;
        }
    }

    free(buffer);
}

int main(int argc, char *argv[]) {
    unsigned long n = 1024;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename> [generations]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int rows = 0;
    int cols = 0;
    int numLiveCells = 0;
    Cell *liveCells = NULL;
    parseMatrix(argv[1], &liveCells, &numLiveCells, &rows, &cols);

    if (rows > n || cols > n) {
        fprintf(stderr, "Matrix dimensions exceed the grid size.\n");
        return EXIT_FAILURE;
    }

    int nGenerations = 1;
    if (argc >= 3) {
        nGenerations = atoi(argv[2]);
    }

    std::cout << "Initial Number of live cells: " << numLiveCells << std::endl;

    // Allocate neighbor counts array
    int gridSize = n * n;
    int *d_neighborCounts;
    cudaMalloc((void**)&d_neighborCounts, gridSize * sizeof(int));
    cudaMemset(d_neighborCounts, 0, gridSize * sizeof(int));

    // Allocate live cells arrays
    Cell *d_liveCells;
    Cell *d_newLiveCells;
    cudaMalloc((void**)&d_liveCells, MAX_LIVE_CELLS * sizeof(Cell));
    cudaMalloc((void**)&d_newLiveCells, MAX_LIVE_CELLS * sizeof(Cell));

    int *d_newNumLiveCells;
    cudaMalloc((void**)&d_newNumLiveCells, sizeof(int));

    // Copy initial live cells to device
    cudaMemcpy(d_liveCells, liveCells, numLiveCells * sizeof(Cell), cudaMemcpyHostToDevice);

    // Timing variables
    struct timespec start, stop;
    double time;

    if (clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime"); }

    for (int gen = 0; gen < nGenerations; ++gen) {
        // Reset neighbor counts
        cudaMemset(d_neighborCounts, 0, gridSize * sizeof(int));

        // Update neighbor counts
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (numLiveCells + threadsPerBlock - 1) / threadsPerBlock;
        updateNeighborCounts<<<blocksPerGrid, threadsPerBlock>>>(d_liveCells, numLiveCells, d_neighborCounts, n);
        cudaDeviceSynchronize();

        // Apply Game of Life rules
        cudaMemset(d_newNumLiveCells, 0, sizeof(int));
        blocksPerGrid = (gridSize + threadsPerBlock - 1) / threadsPerBlock;
        applyGameOfLifeRules<<<blocksPerGrid, threadsPerBlock>>>(d_liveCells, numLiveCells, d_neighborCounts, n, d_newLiveCells, d_newNumLiveCells);
        cudaDeviceSynchronize();

        // Get number of new live cells
        cudaMemcpy(&numLiveCells, d_newNumLiveCells, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap live cells arrays
        Cell *temp = d_liveCells;
        d_liveCells = d_newLiveCells;
        d_newLiveCells = temp;
    }

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1) { perror("clock gettime"); }
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time for %d generations: %f ms\n", nGenerations, time * 1e3);

    // Copy final live cells to host
    liveCells = (Cell*)realloc(liveCells, numLiveCells * sizeof(Cell));
    cudaMemcpy(liveCells, d_liveCells, numLiveCells * sizeof(Cell), cudaMemcpyDeviceToHost);

    // Print the final generation's board
    std::cout << "Final Generation (" << numLiveCells << " live cells):" << std::endl;
    printBoard(liveCells, numLiveCells, n);

    // Clean up
    cudaFree(d_liveCells);
    cudaFree(d_newLiveCells);
    cudaFree(d_neighborCounts);
    cudaFree(d_newNumLiveCells);
    free(liveCells);

    return 0;
}
