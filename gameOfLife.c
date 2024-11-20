#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

void gameOfLife(int* board, int* nBoard, int rows, int cols) {
    // Direction vectors for 8 neighbors
    int dir[8][2] = {{0,1}, {0,-1}, {1,1}, {1,-1}, {1,0}, {-1,0}, {-1,1}, {-1,-1}};
    
    // Update the board based on the rules
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int live_neighbors = 0;

            // Count live neighbors
            for (int k = 0; k < 8; k++) {
                int new_i = i + dir[k][0];
                int new_j = j + dir[k][1];

                // Check if the new position is within bounds
                if (new_i >= 0 && new_i < rows && new_j >= 0 && new_j < cols) {
                    if (board[new_i * cols + new_j] == 1) {
                        live_neighbors++;
                    }
                }
            }

            // Apply the Game of Life rules
            int index = i * cols + j; // Index for the current cell
            if (board[index] == 0 && live_neighbors == 3) {
                nBoard[index] = 1; // Mark as a cell that will become alive
            } 
            if (board[index] == 1) {
                if (live_neighbors < 2 || live_neighbors > 3) {
                   nBoard[index] = 0; // Mark as a cell that will die
                }
            }
        }
    }
}

void printBoard(int* board, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%d ", board[i * col + j]);
        }
        printf("\n");
    }
    printf("\n");
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
    

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int *A;
    int rows = 0;
    int cols = 0;
    parseMatrix(argv[1], &A, &rows, &cols);
    int *B = (int*) malloc(sizeof(int)*rows*cols);
    for (int i=0; i<n; i++){
        for(int j=0; j< n; j++){
            B[i*n + j]=0;  
        }   
    }   

    //printBoard(matrix, rows, cols);
    
    struct timespec start, stop; 
    double time;
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    int nGen = 10;
    for(int i = 0; i < nGen; i++){
        gameOfLife(A, B, n, n);
        memcpy(A, B, sizeof(int)*rows*cols);
    }
    

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
	
    printf("Execution time = %f s\n", time);	
    //printBoard(A, n, n);	

    return 0;
}
