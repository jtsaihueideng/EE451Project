#include <iostream>
#include <cuda.h>
#include <ctime>

#define BLOCK_SIZE 32  // Define BLOCK_SIZE for flexible grid size

__global__ void gameOfLife(int* A, int* B) {
    int n = 1024;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int localRow = threadIdx.y + 1;  // Adjust for shared memory padding
    int localCol = threadIdx.x + 1;

    __shared__ int sharedA[BLOCK_SIZE + 2][BLOCK_SIZE + 2];  // Shared memory with padding for halo cells

    // Initialize shared memory with padding for halos
    if (row < n && col < n) {
        // Center cell
        sharedA[localRow][localCol] = A[row * n + col];
        
        
        if (threadIdx.y == 0 && row > 0) {
            sharedA[0][localCol] = A[(row - 1) * n + col];
        }
        if (threadIdx.y == BLOCK_SIZE - 1 && row < n - 1) {
            sharedA[BLOCK_SIZE + 1][localCol] = A[(row + 1) * n + col];
        }
        if (threadIdx.x == 0 && col > 0) {
            sharedA[localRow][0] = A[row * n + (col - 1)];
        }
        if (threadIdx.x == BLOCK_SIZE - 1 && col < n - 1) {
            sharedA[localRow][BLOCK_SIZE + 1] = A[row * n + (col + 1)];
        }

        // Load corner halo cells only if within bounds
        if (threadIdx.x == 0 && threadIdx.y == 0 && row > 0 && col > 0) {
            sharedA[0][0] = A[(row - 1) * n + (col - 1)];  // Top-left corner
        }           
        if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0 && row > 0 && col < n - 1) {
            sharedA[0][BLOCK_SIZE + 1] = A[(row - 1) * n + (col + 1)];
        }
        if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1 && row < n - 1 && col > 0) {
            sharedA[BLOCK_SIZE + 1][0] = A[(row + 1) * n + (col - 1)]; 
        }
        if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1 && row < n - 1 && col < n - 1) {
            sharedA[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = A[(row + 1) * n + (col + 1)];
        }
    }
    __syncthreads();

    
    if (row < n && col < n) {
        int liveNeighbors = 0;

        // Sum live neighbors, accounting for edges
        if (localRow > 0 && localCol > 0) {
            liveNeighbors += sharedA[localRow - 1][localCol - 1];  // Top-left
        }
        if (localRow > 0) {
            liveNeighbors += sharedA[localRow - 1][localCol];  // Top
        }
        if (localRow > 0 && localCol < BLOCK_SIZE + 1) {
            liveNeighbors += sharedA[localRow - 1][localCol + 1];  // Top-right
        }
        if (localCol > 0) {
            liveNeighbors += sharedA[localRow][localCol - 1];  // Left
        }
        if (localCol < BLOCK_SIZE + 1) {
            liveNeighbors += sharedA[localRow][localCol + 1];  // Right
        }
        if (localRow < BLOCK_SIZE + 1 && localCol > 0) {
            liveNeighbors += sharedA[localRow + 1][localCol - 1];  // Bottom-left
        }
        if (localRow < BLOCK_SIZE + 1) {
            liveNeighbors += sharedA[localRow + 1][localCol];  // Bottom
        }
        if (localRow < BLOCK_SIZE + 1 && localCol < BLOCK_SIZE + 1) {
            liveNeighbors += sharedA[localRow + 1][localCol + 1];  // Bottom-right
        }

        int currentState = sharedA[localRow][localCol];
        int nextState;

        // Game of Life rules
        if (currentState == 1) {
            nextState = (liveNeighbors == 2 || liveNeighbors == 3) ? 1 : 0;
        } else {
            nextState = (liveNeighbors == 3) ? 1 : 0;
        }

        B[row * n + col] = nextState;
    }
}

void printBoard(int* board, int startRow = 0, int startCol = 0, int rows = 10, int cols = 10) {
    int n = 1024;
    for (int i = startRow; i < startRow + rows; ++i) {
        for (int j = startCol; j < startCol + cols; ++j) {
            std::cout << board[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void parseMatrix(const char *filename, int **matrix, int *rows, int *cols) {
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

    printf("Parsing 1D Matrix %d x %d\n", *rows, *cols);

    // Allocate memory for the 1D matrix
    *matrix = (int *)malloc((*rows) * (*cols) * sizeof(int));
    if (!(*matrix)) {
        perror("Error allocating memory for matrix");
        free(buffer);
        exit(EXIT_FAILURE);
    }

    // Second pass: Parse and store values in the matrix
    int index = 0;
    for (int i = 0; buffer[i] != '\0'; ++i) {
        if (isdigit(buffer[i])) {
            (*matrix)[index++] = buffer[i] - '0'; // Convert char to int
        }
    }

    free(buffer);
}



int main(int argc, char *argv[]) {
    unsigned long n = 1024;
    dim3 dimGrid(n/BLOCK_SIZE,n/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int *A;
    int rows = 0;
    int cols = 0;
    parseMatrix(argv[1], &A, &rows, &cols);

    // int *A = (int*) malloc(sizeof(int)*n*n);
    int *B = (int*) malloc(sizeof(int)*n*n);
    
    // Initialize array
    int i,j;
    for (i=0; i<n; i++){
        for(j=0; j< n; j++){
            // A[i*n + j]=i%2;
            B[i*n + j]=0;  
        }   
    }   
    
    int *gpu_A;
    int *gpu_B;
    
    cudaMalloc((void**)&gpu_A, sizeof(int)*n*n);
    cudaMalloc((void**)&gpu_B, sizeof(int)*n*n);
    struct timespec start, stop; 
    double time;
        
    int nGenerations = 1;
    printBoard(A);
    
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    cudaMemcpy(gpu_A, A, sizeof(int)*n*n, cudaMemcpyHostToDevice);
    for (i = 0; i < nGenerations; i++) {      
        gameOfLife<<<dimGrid,dimBlock>>>(gpu_A,gpu_B);
        cudaDeviceSynchronize();
        cudaMemcpy(gpu_A, gpu_B, sizeof(int) * n * n, cudaMemcpyDeviceToDevice);
    }   
    cudaMemcpy(B, gpu_B, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("time is %f ns\n", time*1e9);
    printBoard(B);
        
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    free(A);
    free(B);
}   
