CC=gcc
SRC=sudoku.c sudoku_parser.c sudoku_generator.c sudokusolver.c
HDR=sudoku.h sudoku_parser.h sudoku_generator.h
OBJ=sudoku_parser.o sudoku_generator.o sudokusolver.o sudoku.o

# Possible SIZE parameters are:
# - M32bit  : solve sudokus up to a block size of 32,  e.g. standard sudokus of size 9x9, 16x16, 25x25
# - M64bit  : solve sudokus up to a block size of 64,  e.g. standard sudokus of size 9x9, 16x16, 25x25, 36x36, 49x49, 64x64
# - M128bit : solve sudokus up to a block size of 128, e.g. standard sudokus of size 9x9, 16x16, 25x25, 36x36, 49x49, 64x64, 81x81, 100x100, 121x121
# Option GCC_POPCNT32: use gcc's built in popcound function for 32 bit integers. This parameter has no effect for SIZE M64bit and M128bit.
# Option GCC_POPCNT64: use gcc's built in popcound function for 64 bit integers. This parameter has no effect for SIZE M32bit but may impact M128bit
# Option GCC_POPCNT128: use gcc's built in popcound function for 128 bit integers. This parameter does not affect M32bit and M64bit. If this parameter
# is not defined the current implementation in the code relies on a 64bit popcount function.
# Useful combinations are thus:
# M32bit					: 32 bit gss with own 32bit popcount implementation
# M32bit + GCC_POPCNT32		: 32 bit gss with gcc's builtin 32bit popcount
# M64bit 					: 64 bit gss with own 64bit popcount implementation
# M64bit + GCC_POPCNT64		: 64 bit gss with gcc's builtin 64bit popcount
# M128bit 					: 128 bit gss with own 128bit popcount implementation relying on its own 64bit popcount implementation
# M128bit + GCC_POPCNT64 	: 128 bit gss with own 128bit popcount implementation relying on gcc's builtin 64bit popcount
# M128bit + GCC_POPCNT128 	: 128 bit gss with gcc's builtin 128bit popcount
# Some combinations make no sense:
# M128bit + GCC_POPCNT32
# M128bit + GCC_POPCNT128 + GCC_POPCNT64
# M64bit + GCC_POPCNT32
# M64bit + GCC_POPCNT128

# CFLAGS=-Wall -Og -g -DM128bit -DGCC_POPCNT64 
# CFLAGS=-Ofast -flto -DM32bit -DGCC_POPCNT32 -march=native
# CFLAGS=-Ofast -flto -DM64bit -DGCC_POPCNT64 -march=native

# -flto and Ofast do not seem to make things faster, builtin popcound does
# CFLAGS=-O3 -flto -DM128bit -DGCC_POPCNT64 -march=native
CFLAGS=-O3 -flto -DM128bit -DGCC_POPCNT128 -march=native
# CFLAGS=-Ofast -flto -D$(SIZE) -DGCC_POPCNT -march=native
LFLAGS=-lm
# CFLAGS=-Ofast -flto -D$(SIZE) -DGCC_POPCNT -march=native
# CFLAGS=-O3 -D$(SIZE) -DGCC_POPCNT -march=native

bindir=/usr/bin
mandir=/usr/share/man/man1
all:gss jigsawmrf
jigsawmrf: JigSawMRF.c
	$(CC) $(CFLAGS) JigSawMRF.c -o JigSawMRF $(LFLAGS) 
gss: $(OBJ) Makefile
	$(CC) $(CFLAGS) -o gss $(OBJ) $(LFLAGS) 
test: gss
	echo Testing gss against the sudoku17 database
	./gss -s solved.dat -c sudoku17_spl.dat >/dev/null
	diff solved.dat ref17.dat > test.log
	rm solved.dat
	rm test.log
	echo Passed
sudoku.o: sudoku.c sudoku.h Makefile
sudoku_parser.o:sudoku_parser.c sudoku.h Makefile
sudoku_generator.o: sudoku_generator.c sudoku.h Makefile
sudokusolver.o: sudokusolver.c sudoku.h Makefile
install: gss 
	install gss $(bindir)
	install JigSawMRF $(bindir)
clean:
	rm *.o gss JigSawMRF
