#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void gameOfLife(int* board, int rows, int cols) {
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
                    if (board[new_i * cols + new_j] == 1 || board[new_i * cols + new_j] == -2) {
                        live_neighbors++;
                    }
                }
            }

            // Apply the Game of Life rules
            int index = i * cols + j; // Index for the current cell
            if (board[index] == 0 && live_neighbors == 3) {
                board[index] = -1; // Mark as a cell that will become alive
            } 
            if (board[index] == 1) {
                if (live_neighbors < 2 || live_neighbors > 3) {
                    board[index] = -2; // Mark as a cell that will die
                }
            }
        }
    }
    // Finalize board updates
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = i * cols + j; // Index for the current cell
            if (board[index] == -1) {
                board[index] = 1; // Set the cell to alive
            }
            if (board[index] == -2) {
                board[index] = 0; // Set the cell to dead
            }
        }
    }
}

void printBoard(int* board, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%d ", board[i * row + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    unsigned long n = 1024;
    int *A = (int*) malloc(sizeof(int)*n*n);
    
    // Initialize array
    int i,j;
    for (i=0; i<n; i++){
        for(j=0; j< n; j++){
            A[i*n + j]=i%2;
        }   
    }

    struct timespec start, stop; 
    double time;
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

    gameOfLife(A, n, n);

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	
    printf("Execution time = %f sec\n", time);	
    //printBoard(A, n, n);	

    return 0;
}
