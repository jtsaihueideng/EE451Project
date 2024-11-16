# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -O2

# Targets
all: serial

# Target to compile p1a.c (naive matrix multiplication)
serial: gameOfLife.c
	$(CC) $(CFLAGS) -o serial gameOfLife.c

clean:
	rm -f serial
