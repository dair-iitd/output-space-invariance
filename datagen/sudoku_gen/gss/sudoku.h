/*
    Generic Sudoku Solver (gss)
    Copyright (C) 2019 B. Pieters

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
/* ************************************************************************************ */
/* I use integers as bit masks where 1 int can represent all possible values of a field */
/* The size of the mask int thus determines the maximum blocksize gss supports          */
/* The size is determined duing compilation. The following sizes are supported          */
/* ************************************************************************************ */
/* 32  bit --> define M32bit during compilation / default option                        */
/* 64  bit --> define M64bit during compilation                                         */
/* 128 bit --> define M128bit during compilation                                        */
/* ************************************************************************************ */
#include <stdint.h>
#ifdef M32bit
	#define NEEDPOPCNT32
	#define MASKINT uint32_t
#elif M64bit
	#define NEEDPOPCNT64
	#define MASKINT uint64_t
#elif M128bit 
	#define NEEDPOPCNT128
	#define MASKINT unsigned __int128
#endif

#ifdef GCC_POPCNT32
	#define popcount32 __builtin_popcount
#endif
#ifdef GCC_POPCNT64
	#define popcount64 __builtin_popcountl
#endif
#ifdef GCC_POPCNT128
	#define popcount64 __builtin_popcountll
#endif

#ifndef MASKINT
#define M32bit
#define MASKINT uint32_t
#endif

/* strategies */
typedef enum STRATEGIES {MASK, MASKHIDDEN, MASKINTER, BRUTE} STRATEGIES;

/* sudoku struct */
typedef struct Sudoku {
	MASKINT *M;					/* sudoku mask (i.e. binary values indicating all possible values of a field)*/
	int N, BS;						/* filed size N and block size BS */
	char *format;						/* format strings to pretty print the sudoku */	
	char uk;						/* character for unknowns */	
	int *Npos;						/* positions in the format strings to place the numbers (array lenth N+1)*/ 
	int **BLKS;						/* sudoku blocks (in a standard sudoku these are rows, columns amd 9 squares) defined for flexibility to solve various flavours of sudokus */
	int **IBLKS;					/* inverse blocks, for each element this links to all the blocks it is a member of */
	int **LINKS;					/* element links */
	int NBLK;						/* number of blocks, for a standard sudoku this is 9 rows + 9 columns + 9 squares = 27 blocks */
	int *lvl1;						/* keep track of level1 eliminated elements */
} Sudoku;

typedef enum SSTATE {UNSOLVED, SOLVED, MULTISOL, INVALID} SSTATE;

extern MASKINT VX[8*sizeof(MASKINT)];					/* binary symbols for the numbers */
void S_init();							/* initialize the VX array */
void S_free(Sudoku *S);						/* clean up a sudoku */
int S_UK(Sudoku S);						/* returns the number of elements that are not known (yet) */
int S_BUK(Sudoku S, int B);
int Check(Sudoku *S);
int * RowBlock(Sudoku S, int i);				/* generate the i-th row block */
int * ColBlock(Sudoku S, int i);				/* generate the i-th column-block */
int * Block(Sudoku S, int i);					/* generate the i-th block-block */
int * DiagBlock(Sudoku S, char d);				/* generate a diagonal block */
void PopulateIBlocks(Sudoku *S);					/* populate the inverse block lookup table */
void S_Print(Sudoku S, FILE *f);						/* dump a sudoku to the screen */
int LogicSolve(Sudoku *S, int *maxlevel, int limitlevel, int STRAT);			/* solve by eliminating */
int BruteForce(Sudoku *S, int *NS, MASKINT ***sol, int maxsol, int limitlevel, int STRAT);	/* in case we are not smart enough we try to do it by brute force (backtracking)*/
SSTATE Solve(Sudoku *S, int *maxlevel, MASKINT ***sol, int *ns, int maxsol, int limitlevel, int STRAT); /* solve a sudoku */
int EL_V(MASKINT V, int BS);						/* returns the human readyble value (1-9) of an element, 0 if the element is unknown V is in the binary representation of all possible vcalues*/
int EL_P(MASKINT V);
int FillEmptySudoku(Sudoku *S);
void ClearSudoku(Sudoku *S);
int ResolveConflicts(Sudoku *S);

extern int GUESS;							/* stores the number of guesses made during solving */

#define SDO(S) (STRAT&(1<<S))				/* macro to test whether a strategy is enabled or not, assumes the variable STRAT is defined!*/
