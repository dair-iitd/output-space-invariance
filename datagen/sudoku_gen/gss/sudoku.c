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
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include "sudoku.h"

#ifdef NEEDPOPCNT32
#ifndef GCC_POPCNT32
static int popcount32 (MASKINT V)
{
	const MASKINT m1  = 0x55555555;
	const MASKINT m2  = 0x33333333;
	const MASKINT m4  = 0x0f0f0f0f;
	const MASKINT h01 = 0x01010101;
	V -= (V >> (MASKINT)1) & m1;            
	V = (V & m2) + ((V >> (MASKINT)2) & m2); 
	V = (V + (V >> (MASKINT)4)) & m4;   
	return (int) ((V * h01)>>(MASKINT)24);	
}
#endif
#endif

#ifdef NEEDPOPCNT128

#ifndef GCC_POPCNT64
static int popcount64 (uint64_t V)
{
	const uint64_t m1  = 0x5555555555555555;
	const uint64_t m2  = 0x3333333333333333;
	const uint64_t m4  = 0x0f0f0f0f0f0f0f0f;
	const uint64_t h01 = 0x0101010101010101;
	V -= (V >> (uint64_t)1) & m1;            
	V = (V & m2) + ((V >> (uint64_t)2) & m2); 
	V = (V + (V >> (uint64_t)4)) & m4;   
	return (int) ((uint64_t)(V * h01)>>(uint64_t)56);
}
#endif

static int popcount128 (MASKINT V)
{
	return popcount64(V)+popcount64(V>>64);
}
#endif

#ifdef NEEDPOPCNT64
#ifndef GCC_POPCNT64
static int popcount64 (MASKINT V)
{
	const MASKINT m1  = 0x5555555555555555;
	const MASKINT m2  = 0x3333333333333333;
	const MASKINT m4  = 0x0f0f0f0f0f0f0f0f;
	const MASKINT h01 = 0x0101010101010101;
	V -= (V >> (MASKINT)1) & m1;            
	V = (V & m2) + ((V >> (MASKINT)2) & m2); 
	V = (V + (V >> (MASKINT)4)) & m4;   
	return (int) ((MASKINT)(V * h01)>>(MASKINT)56);
}
#endif
#endif

/* an attempt to write code that solves generic sudokus puzzels by logic rather than trial and error */
/* if the code is not smart enough it will do the trial and error thing as a last resort */
/* Parts of the algorithm are based on that of  Kurt Carlson, fnkac@uaf.edu or kcarlson@arsc.edu*/

int GUESS;							/* count number of guesses in a puzzle */

MASKINT VX[8*sizeof(MASKINT)];		/* binary symbols for the numbers */
void TestPOP();
void S_init()				/* initialize the VX array */
{
	int i;
	for (i=0;i<(int)(8*sizeof(MASKINT));i++)
		VX[i]=((MASKINT)1<<(MASKINT)i);
	TestPOP();
	
}

void S_free(Sudoku *S)	/* clean up a sudoku */
{
	int i;
	free(S->M);
	free(S->Npos);
	free(S->format);
	for (i=0;i<S->NBLK;i++)
		free(S->BLKS[i]);
	free(S->BLKS);
	for (i=0;i<S->N;i++)
	{
		free(S->IBLKS[i]);
		free(S->LINKS[i]);
	}
	free(S->IBLKS);
	free(S->LINKS);
	free(S->lvl1);
	S->NBLK=0;	
}

void PrintB(MASKINT B)
{
	MASKINT mask=1;
	int i;
	mask<<=8*sizeof(MASKINT)-1;
	for (i=0;i<(int)(8*sizeof(MASKINT));i++)
	{
		if (mask&B)
			putchar('1');
		else
			putchar('0');
		
		mask>>=1;
	}
	
	
}
/************************************** Access Sudoku Data *************************************************/
int EL_V(MASKINT V, int BS)
/* given a bitmask return the value of a field. 
 * If more than one value is possible, return 0
 */
{
	int i;
	
	for (i=0;i<BS;i++)
		if (V==VX[i])
			return i+1;
	return 0;
	
}
#ifdef M128bit
int EL_P(MASKINT V)
/* returns the number possible values for an 
 * element (is the binary representation of 
 * all possible values). If GCC_POPCNT is 
 * defined this function refers to
 * __builtin_popcount. Here we make our own 
 * implementation of popcount in case the 
 * builtin function is not available.
 */
{
	return popcount128(V);
}
#endif

#ifdef M64bit
int EL_P(MASKINT V)
/* returns the number possible values for an 
 * element (is the binary representation of 
 * all possible values). If GCC_POPCNT is 
 * defined this function refers to
 * __builtin_popcount. Here we make our own 
 * implementation of popcount in case the 
 * builtin function is not available.
 */
{
	return popcount64(V);
}
#endif

#ifdef M32bit
int EL_P(MASKINT V)		
{
	return popcount32(V);
}
#endif

void TestPOP()
{
	int i;
	MASKINT M, N=0;
	for (i=0;i<8*sizeof(MASKINT);i++)
	{
		M=(MASKINT)1<<(MASKINT)i;
		N|=M;
		if (EL_P(M)!=1)
		{
			fprintf(stderr, "Error: dysfunctional popcount detected\n");
			fprintf(stderr, "popcount %u should be 1, %d\n",M,EL_P(M));
			exit(1);
		}
	}
	if (EL_P(N)!=i)
	{
		fprintf(stderr, "Error: dysfunctional popcount detected\n");
		fprintf(stderr, "popcount %u should be %d\n",N,i);
		exit(1);
	}
}

int S_UK(Sudoku S)
/* Determine the number of unsolved 
 * elements in the sudoku
 */
{
	int i, r;
	r=0;
	for (i=0;i<S.N;i++)
		r+=(EL_P(S.M[i])>1);
	return r;
		
}

int S_BUK(Sudoku S, int B)
/* Determine the number of unsolved 
 * elements in block B
 */
{
	int i, r;
	r=0;
	if (B>=S.NBLK)
		return 0;
	for (i=0;i<S.BS;i++)
		r+=(EL_P(S.M[S.BLKS[B][i]])>1);
	return r;
		
}


/************************************** Utillities to Generate Sudokus *************************************************/
/* as this is a generic sudoku solver we need some routines to initialize the structure of the sudoku.
 * In particular we need to define "blocks". In a standard sudoku the blocks would be the rows, columns, 
 * and squares. All blocks must have the same size (typ. 9 elements) and blocks overlap, i.e. elements
 * may be a member of more than one block (typ. each element is a member of 3 blocks, row, column, and 
 * square).
 */

int * RowBlock(Sudoku S, int i)
/* create the standard i-th rowblock */
{
	int *R;
	int k=0;
	R=malloc(S.BS*sizeof(MASKINT));
	if (S.N%S.BS)
	{
		fprintf(stderr, "Error: not all rows are %d in size, cannot define row blocks\n", S.BS);
		exit(1);
	}
	i*=S.N/S.BS;
	for (k=0;k<S.BS;k++)
		R[k]=i+k;
	return R;
}

int * ColBlock(Sudoku S, int i)
/* create the standard i-th columnblock */
{
	int *R;
	int k=0, RL;
	R=malloc(S.BS*sizeof(int));
	RL=S.N/S.BS;
	if  ((S.N%RL)&&(S.N/RL==S.BS))
	{
		fprintf(stderr, "Error: not all columns are %d in size, cannot define column blocks\n", S.BS);
		exit(1);
	}
	for (k=0;k<S.BS;k++)
		R[k]=i+k*RL;
	return R;
}

int * Block(Sudoku S, int i)
/* create the standard i-th squareblock */
{
	int *R;
	int j, ll;
	int k,l, z=0;

	for (j=2;j*j<=(int)(8*sizeof(MASKINT));j++)
	{
		int N, BS;
		BS=j*j;
		N=BS*BS;
		if ((S.N==N)&&(S.BS==BS))
			break;
	}
	
	if (j*j>(int)(8*sizeof(MASKINT)))
	{
		fprintf(stderr, "Error: Standard blocks only works for standard sized sudokus\n");
		exit(1);
	}
	ll=j;
	R=malloc(S.BS*sizeof(int));
	j=i%ll;
	i=i/ll;
	for (k=i*ll;k<i*ll+ll;k++)
		for (l=j*ll;l<j*ll+ll;l++)
			R[z++]=k*S.BS+l;
	return R;
}

int * DiagBlock(Sudoku S, char d)
/* create a down-diagonal block */
{
	int issq;
	int *R;
	int k=0, off=1, s=0;
	R=malloc(S.BS*sizeof(MASKINT));
	issq=S.N/S.BS;
	
	if ((S.N-issq*issq!=0)||(S.N%S.BS))
	{
		fprintf(stderr, "Error: not all need a square susoku with %d elements (for a %d sized sudoku) along the diagonal\n", S.BS, S.N);
		exit(1);
	}
	if (d=='d')
	{
		s=0;
		off=1;
	}
	else if (d=='u')
	{
		s=S.BS-1;
		off=-1;
	}
	for (k=0;k<S.BS;k++)
		R[k]=s+k*(S.BS+off);
	return R;
}
/* to allow the efficient solution we precompute a set of tables and store it in the sudoku data structure */

int IsInSet(int i, int set[], int n)
/* utillity function, given a set of n numbers in set, is i a member? */
{
	int j;
	for (j=0;j<n;j++)
		if (i==set[j])
			return 1;
	return 0;
}

int IsSubSet(int *B, int  set[], int level, int BS)
/* *****function is obsolete******
 * given a set of level numbers in B, is B a subset of the BS numbers in set */
{
	int i, j;
	for (i=0;i<level;i++)
	{
		for (j=0;j<BS;j++)
			if (B[j]==set[i])
				break;
		if (j==BS)
			return 0;
	}
	return 1;
}


int *AddBlock(int el, int *list, int *Block, int BS)
/* given an element el, and a block Block, add all elements of Block
 * except for el to the list list. After calling this routine for one 
 * element with all blocks in the sudoku, the array list whould contain 
 * all elements element el is directly "linked" to
 */
{
	int i;
	for (i=0;i<BS;i++)
	{
		if (Block[i]!=el)
			if (!IsInSet(Block[i], list+1, list[0]))
			{
				if ((list[0]+2)%BS==0)
					list=realloc(list,(list[0]+BS+1)*sizeof(int));
				list[list[0]+1]=Block[i];
				list[0]++;
			}
	}
	return list;		
}

void PopulateIBlocks(Sudoku *S)
/* here we compute two tables:
 * IBLKS: for each element we store an array of boolean 
 * values listing what blocks the element is a member of
 * LINKS: for each element list what elements it is "linked"
 * to. one LINK array for one element is an int pointer where 
 * the first element is the number of "links" and the rest 
 * are the linked elements, i.e. the total length is the 
 * number of elements the element is linked to + 1 
 */
{
	/* IBLKS should be an array of S.N long, which for each element holds S.NBLK integers */
	int i,j;
	for (i=0;i<S->N;i++)
	{
		S->LINKS[i][0]=0; /* clear LINKS */
		for (j=0;j<S->NBLK;j++)
			S->IBLKS[i][j]=0; /* clear IBLKS */
	}
			
	for (i=0;i<S->NBLK;i++)
		for (j=0;j<S->BS;j++)
			S->IBLKS[S->BLKS[i][j]][i]=1;
			
	for (i=0;i<S->N;i++)
	{
		int b=0;
		for (j=0;j<S->NBLK;j++)
		{
			/* add elements from block j to the link list of element i */
			if (S->IBLKS[i][j])
			{
				b++;
				S->LINKS[i]=AddBlock(i, S->LINKS[i], S->BLKS[j], S->BS);
			}
		}
		if (!b)
		{
			fprintf(stderr, "Error: not all elements are members of a block!\n");
			exit(1);
		}
	}
}


void S_Print(Sudoku S, FILE *f)
/* dump a sudoku to file f */
{
	int i=0,v, ds;
	int p=0;
	ds=1;
	i=S.BS;
	while (i/=10)
		ds++;
	do {
		while ((S.Npos[i]==p)&&(i<S.N))
		{
			v=EL_V(S.M[i], S.BS);
			if (v)
				fprintf(f, "%0*d", ds, v);
			else
			{
				for (v=0;v<ds;v++)
					fprintf(f,"%c", S.uk);
			}
			i++;
		}
		if (S.format[p])
			fputc(S.format[p],f);
		p++;
	} while (S.format[p]);
}

void S_Print_Mask(Sudoku S)
/* print the bitmask to the stdout */
{
	int i=0;
	int p=0;
	do {
		while ((S.Npos[i]==p)&&(i<S.N))
		{
			printf("%lu",S.M[i]);
			i++;
		}
		if (S.format[p])
			putchar(S.format[p]);
		p++;
	} while (S.format[p]);
}
/************************************** Utillities to Solve Sudokus *************************************************/


/* use the twiddle algorithm for iterate through all combinations of M of N elements */
/*  Reference for the twiddle algorithm:
  Phillip J Chase, `Algorithm 382: Combinations of M out of N Objects [G6]',
  Communications of the Association for Computing Machinery 13:6:368 (1970). */

int twiddle (int *X, int *Y, int *P)
/* core of the twiddle algorithm */
{
	int i, j, k;
	j = 1;
	while(P[j] <= 0)
		j++;
	if(P[j-1] == 0)
	{
		for(i = j-1; i != 1; i--)
			P[i] = -1;
		P[j] = 0;
		*X = 0;
		P[1] = 1;
		*Y = j-1;
	}
	else
	{
		if(j > 1)
			P[j-1] = 0;
		do
			j++;
		while(P[j] > 0);
		k = j-1;
		i = j;
		while(P[i] == 0)
			P[i++] = -1;
		if(P[i] == -1)
		{
			P[i] = P[k];
			*X = i-1;
			*Y = k-1;
			P[k] = -1;
		}
		else
		{
			if(i == P[0])
				return(1);
			else
			{
				P[j] = P[i];
				P[i] = 0;
				*X = j-1;
				*Y = i-1;
			}
		}
	}
	return(0);
}


int FirstCombo(int el[], int *set, int *P, int N, int M)
/* determine the first combination of M 
 * elements in a set of N and initialize the P array
 * el: set of N elements
 * set: subset of M elements
 * P: array of N+1 elements
 * if a combination of M elements of N elements is possible 
 * (i.e. M<=N), return 1, else 0
 */
{
	int i;
	if (M>N)
		return 0;
	for (i=0;i<M;i++)
		set[i]=el[N-M+i];
	P[0] = N+1;
	for(i = 1; i != N-M+1; i++)
		P[i] = 0;
	while(i != N+1)
	{
		P[i] = i+M-N;
		i++;
	}
	P[N+1] = -2;
	if(M == 0)
		P[1] = 1;
	return 1;
	
}

int SetIndex(int set[], int v, int M)
/* find the index of element v in a set set, 
 * if the element is not a member it returns 
 * the number of elemnts in the set
 */
{
	int i=0;
	while ((set[i]!=v)&&(i<M))
		i++;
	return i;
}

int NextCombo(int el[], int *set, int *P, int N, int M)
/* Determine the next combination of M 
 * elements in a set of N
 * If there is no combination left
 * return 0, else 1
 * Before using this routine the first 
 * subset must be computed using 
 * FirstCombo, which also initilized 
 * the P array
 */
{
	static int x, y, z;
	if (!twiddle (&x, &y, P))
	{
		z=SetIndex(set, el[y], M);
		set[z]=el[x];
		return 1;
	}
	return 0;
	
}



int Eliminate(Sudoku *S, int el)
/* basic, recursive, elimination.
 * If an elemnt is known use the LINK 
 * table to eliminate the element value 
 * from all linked elements
 * If this in turn leads to one of the 
 * linked elements having only one 
 * possible value left, call Eliminate 
 * recursively with this element
 */
{
	MASKINT M;
	int elim=0,r;
	int j;
	if (S->lvl1[el])
		return 0;
	if(EL_P((S->M[el]))==1)
	{
		S->lvl1[el]=1;	
		M=~(S->M[el]);
		for (j=1;j<=S->LINKS[el][0];j++)
		{
			int pre, post;
			pre=EL_P(S->M[S->LINKS[el][j]]);
			(S->M[S->LINKS[el][j]])&=M;
			post=EL_P(S->M[S->LINKS[el][j]]);
			elim+=pre-post;
			if (post==1) /* new element solved, recurse */
			{
				r=Eliminate(S, S->LINKS[el][j]);
				if (r<0)
					return r;
				else
					elim+=r;
			}
			else if (post==0)/* eliminated all possibilities for an element, i.e. the sudoku is singular */
				return -1;
		}		
	}
	return elim;	
}


int FastLevel1(Sudoku *S)
/* the first step in solving a sudoku: 
 * go through all elements and if the 
 * element is known call Eliminate on it
 * We only go through all elements once.
 * Along the further solving procedure 
 * whenever we solve one element it 
 * suffices to call the the recursive 
 * Eliminate.  
 */
{
	int i;
	int elim=0,r;
	for (i=0;i<S->N;i++)
	{
		r=Eliminate(S,i);
		if (r<0)
			return r;
		else
			elim+=r;
	}	
	return elim;
}

int FastLevel2Set(Sudoku *S)
/* fast set elimination for level 2 */
{
	int i,j, k, l;
	int elim=0;
	int *el, n;
	int *B;
	elim=0;
	MASKINT AM=0;
	el=malloc(S->BS*sizeof(int));
	for (i=0;i<S->NBLK;i++)
	{
		B=S->BLKS[i];
		/* find possible set */
		n=0;
		for (j=0;j<S->BS;j++)	/* select all elements with up to level possible values */
		{
			if (EL_P(S->M[B[j]])==2)
				el[n++]=j;
		}
		if (n<2)
			continue;
		for (j=0;j<n;j++)
			for (k=j+1;k<n;k++) /* all combinations of two elements */
				if (S->M[B[el[j]]]==S->M[B[el[k]]]) /* match */
				{
					AM=~S->M[B[el[j]]];
					for (l=0;l<S->BS;l++)
					{
						if ((l!=el[j])&&(l!=el[k]))
						{
							int pre, post, r;
							pre=EL_P((S->M[B[l]]));
							(S->M[B[l]])&=AM;
							post=EL_P((S->M[B[l]]));
							elim+=pre-post;		
							if (post==1)
							{
								r=Eliminate(S, B[l]);
								if (r<0)
								{
									free(el);
									return r;
								}
								else
									elim+=r;
							}
							
						}
					}
				}
		
	}
	free(el);
	return elim;	
}


int SetEliminate(Sudoku *S, int level)
/* More advanced elimination, elimination of a set:
 * If we have a set of n elements in a block, whose 
 * only possible values are limited to a set of n 
 * possible values, we can eliminate these n values 
 * from all other elements within the block.
 * for example, if we find 3 elements with the 
 * following possible values:
 * el1: 1,2,3
 * el2: 1,2
 * el3: 2,3
 * we can eliminate 1,2, and 3 from elements 4-9 
 * The level refers to the size of the set 
 * (i.e. level=n). Level is thus a measure of 
 * the complexity of this elimination method.
 * Note that level 1 reduces to ordinary 
 * elimination.
 */
{
	int *el, n=0;
	int i,j, l;
	MASKINT AM;
	MASKINT M;
	int *set;
	int elim=0;
	int *B;
	int *P;
	int *s_set;
	int MinLevel;
	if (level==1)
		return FastLevel1(S);
	if (level==2)
		return FastLevel2Set(S);
		
	el=malloc(S->BS*sizeof(int));
	set=malloc(S->BS*sizeof(int));
	s_set=malloc(S->BS*sizeof(int));
	P=malloc((S->BS+2)*sizeof(int));
	MinLevel=(level>1);		/* we only select elements with only one possible value if we do a level one elimination */
	for (l=0;l<S->NBLK;l++) /* could we parallellize this?, we could collect data of what we can eliminate for all blocks in parallel and then serially and recuresively eliminate */
	{
		B=S->BLKS[l];
		n=0;
		AM=0;
		
		for (i=0;i<S->BS;i++)	/* select all elements with up to level possible values */
		{
			j=EL_P((S->M[B[i]]));
			if (j>MinLevel)
			{
				AM|=S->M[B[i]]; /* all possibilities in this block excluding solved ones */
				if (j<=level)
				{
					el[n]=i;
					n++;
				}
			}
		}
		/* we now have to find level elements among these which share the same set of possibilities */
		if (n<level)		/* not possible */
			continue;
		
		FirstCombo(el, set, P, n, level); /* we iterate through all possible combinations of level elements from el */ 
		do
		{
			M=0;
			for (j=0;j<level;j++)
			{
				s_set[j]=B[set[j]];
				M|=(S->M[s_set[j]]);
			}
			
			if ((EL_P(M)==level)&&((~M)&AM)) /* level options in set *and* perhaps we can eliminate somthing */
			{
				M=~M;
				for (j=0;j<S->BS;j++)
					if (!IsInSet(B[j],s_set,level))
					{
						if ((S->M[B[j]])&M)
						{
							int pre, post, r;
							pre=EL_P((S->M[B[j]]));
							(S->M[B[j]])&=M;
							post=EL_P((S->M[B[j]]));
							elim+=pre-post;		
							if (post==1)
							{
								r=Eliminate(S, B[j]);
								if (r<0)
								{
									free(el);
									free(set);
									free(s_set);
									free(P);
									return r;
								}
								else
									elim+=r;
							}
													
						}
						else
						{
							/* eliminated all possibilities for an element, i.e. the sudoku is singular */
							(S->M[B[j]])&=M; /* so we know where the singularity is */							
							free(el);
							free(set);
							free(s_set);
							free(P);
							return -1;
						}
					}
			}
		}while(NextCombo(el, set, P, n, level));
	}
	free(el);
	free(set);
	free(s_set);
	free(P);
	return elim;	
}


int FastLevel1Hidden(Sudoku *S)
/* More advanced elimination, elimination of a hidden single:
 * if in a block one particular value is only possible
 * in one particular element, this element must have 
 * that particular value. e.g:
 * el1: 1,2,3,4,5,6,7,8,9
 * el2: 2,3,4,5,6,7,8,9
 * el3: 2,3,4,5,6,7,8,9
 * el4: 2,3,4,5,6,7,8,9
 * el5: 2,3,4,5,6,7,8,9
 * el6: 2,3,4,5,6,7,8,9
 * el7: 2,3,4,5,6,7,8,9
 * el8: 2,3,4,5,6,7,8,9
 * el9: 2,3,4,5,6,7,8,9
 * -> el1 must be 1
 */ 
{
	int i,l,r, pre, post;
	int elim=0;
	MASKINT M1, M2, M;
	int *B;
		
	for (l=0;l<S->NBLK;l++)
	{
		B=S->BLKS[l]; 
		M1=0;
		M2=0;
		for (i=0;i<S->BS;i++)
		{
			if (EL_P(S->M[B[i]])==1)
				M2|=S->M[B[i]];	/* solved elements are ignored */
			else
			{
				M2|=(M1&S->M[B[i]]); /* the not unique bits */
				M1|=S->M[B[i]]; /* occurs at least once */
			}
		}
		M=M1&~M2; /* numbers that occur only once */
		i=0;
		while (M)
		{
			if (M&S->M[B[i]])
			{
				pre=EL_P((S->M[B[i]]));
				S->M[B[i]]&=M;
				M&=~S->M[B[i]];
				post=EL_P((S->M[B[i]]));
				if (post==0)
					return -1;					
				elim+=pre-post;
				if (post==1)
				{
					r=Eliminate(S, B[i]);
					if (r<0)
						return r;
					else
						elim+=r;
				}
			}
			i++;
			if (i==S->BS)
				break;
		}
	}
	return elim;
}

int SetEliminateHidden(Sudoku *S, int level)
/* More advanced elimination, elimination of a hidden set:
 * Like for the hidden single, however, now with a set 
 * of n values
 * e.g. if the set [1,2] is unique to 2 elements, these two elements
 * can only be 1 or 2 and other possible values can be eliminated 
 * from these two elements. For example if we have
 * el1: 1,2,3,4,5
 * el2: 1,2,7,8,9
 * we can eliminate 3,4,5 from el1 and 7,8,9 from el2
 * This routine extends this to sets with n=level values
 */
{
	int *el, n=0;
	int i,j, l;
	MASKINT M, N, UK;
	int *set;
	int elim=0;
	int *B;
	int *P;
	el=malloc(S->BS*sizeof(int));
	set=malloc(S->BS*sizeof(int));
	P=malloc((S->BS+2)*sizeof(int));
	UK=0;
	for (i=0;i<S->BS;i++)
		UK|=VX[i];
	for (l=0;l<S->NBLK;l++)
	{
		B=S->BLKS[l];
		n=0;
		
		/* look for n symbols exclusively present in n elements within a block */ 
		for (i=0;i<S->BS;i++)	/* select all elements with level or more possible values */
		{
			if (EL_P((S->M[B[i]]))>level) /* has to be more than level, there has to be somethng to eliminate */
			{
				el[n]=i;
				n++;
			}
		}
		/* we now have to find level elements among these which have an exclusive subset of level elements */
		if (n<=level)		/* not possible */
			continue;
		
		FirstCombo(el, set, P, n, level); /* we iterate through all possible combinations of level elements from el */ 
		do
		{
			M=0;
			N=0;
			for (j=0;j<S->BS;j++)
			{
				if (IsInSet(j, set, level))
					M|=(S->M[B[j]]);
				else
					N|=(S->M[B[j]]);
			}
			if ((M|N)!=UK) /* cheap position to check consistency of the sudoku as the data is available */
			{
				free(el);
				free(set);
				free(P);
				return -1;
			}
			M&=(~N); /* mask of all symbols exclusive to the current set of elements */
			if (EL_P(M)==level)
			{
				/* we found a full set! */
				for (j=0;j<level;j++)
				{
					int pre,post,r;
					pre=EL_P((S->M[B[set[j]]]));
					S->M[B[set[j]]]&=M;
					post=EL_P((S->M[B[set[j]]]));
					elim+=pre-post;
					if (post==1)
					{
						r=Eliminate(S, B[set[j]]);
						if (r<0)
						{
							free(el);
							free(set);
							free(P);
							return r;
						}
						else
							elim+=r;
					} else if (post==0)
					{
						free(el);
						free(set);
						free(P);
						return -1;
					}
				}
			}
		}while(NextCombo(el, set, P, n, level));
	}
	free(el);
	free(set);
	free(P);
	return elim;	
}

int EliminateInterBlock(Sudoku *S, int level)
/* More advanced elimination, block interaction
 * If we have a set of n elements in block1, which
 * are the only elements within block1 that may 
 * contain a certain value, AND this set of elements
 * in block1 is also subset of elements from block2, 
 * we can eliminate value from elements in block2 which 
 * are not in the subset.
 * 
 * Note that this still does not include more 
 * complicated inter-block links, like X-wing and such.
 * wich involves more blocks
 * I suppose what we should do is:
 * 1: find elements where blocks overlap
 * 2: identify linked pairs, e.g., if el 1 is 7 el2 is not and vise versa
 * 3: see if the linked pairs are again a subset of some block */
{
	int i, j, k, n;
	MASKINT M;
	int *set;
	int blk;
	int elim=0;
	
	/* find sets of elements sharing one specific number and see it it overlaps with other blocks */
	/* we should extend this to up to level-1 values */
	set=malloc(S->BS*sizeof(int));
	for (k=0;k<S->NBLK;k++)
	{
		for (i=0;i<S->BS;i++)
		{
			/* element VX[i] */
			n=0;
			for (j=0;j<S->BS;j++)
			{
				if (VX[i]&(S->M[ S->BLKS[k][j]]))
				{
					set[n]=S->BLKS[k][j];
					n++;
					if (n>level)
						break;
				}
			}
			if (n==level)
			{
				M=~VX[i];
				/* find out whether the set is a subset of another block, if so mask the other block from VX[i] */
				for (j=0;j<S->NBLK;j++)
				{
					if (j==k)
						continue;
					blk=1;
					n=0;
					while ((blk)&&(n<level))
						if(!S->IBLKS[set[n++]][j])
							blk=0;
					
					if (blk) /* the set is also a subset of block j */
					{
						for (n=0;n<S->BS;n++)
							if (!IsInSet(S->BLKS[j][n], set, level))
							{
								int pre,post,r;
								pre=EL_P((S->M[S->BLKS[j][n]]));
								(S->M[ S->BLKS[j][n]])&=M;
								post=EL_P((S->M[S->BLKS[j][n]]));
								elim+=pre-post;
								if (post==1) 
								{
									r=Eliminate(S, S->BLKS[j][n]);
									if (r<0)
									{
										free(set);
										return r;
									}
									else
										elim+=r;
								} else if (post==0)
								{
									free(set);
									return -1;
								}
							}
					}
				}
			}
		}
	}
	
	free(set);
	
	return elim;	
}


int Check(Sudoku *S)
/* Check whether all numbers occur in each block */
{
	MASKINT M, UK;
	int i, j;
	UK=0;
	for (i=0;i<S->BS;i++)
		UK|=VX[i];
	for (i=0;i<S->NBLK;i++)
	{
		M=0;
		for (j=0;j<S->BS;j++)
			M|=S->M[S->BLKS[i][j]];
		if (M!=UK)
			return 1;
	}
	return 0;
}


int LogicSolve(Sudoku *S, int *maxlevel, int limitlevel, int STRAT)
/* solve sudoku's with logic */
{
	int v, level=1, elim=1, err;
	(*maxlevel)=1;	

	if ((limitlevel<=0)||(limitlevel>=S->BS))
		limitlevel=S->BS-1;
		
	while (S_UK(*S))
	{	
		if (elim==0) /* if we did not eliminate anything during the last round, try a higher level */
			level++;
		else
			level=1;
		if (level>limitlevel) /* alas, it did not work, our logic is too primitive */
			break;		
		if (level>(*maxlevel))
			(*maxlevel)=level;
		elim=0;
		if (level==1)
		{
			err=FastLevel1Hidden(S);
			if (err>=0)
				elim+=err;
			else
				return err;
		}
		else
		{
			/* these are the more complicated strategies working on larger sets of elements
			 * For standard sudoku's (9x9) these are usually quite expensive compared to backtracking
			 * however, as the sudokus gets larger backtracking just scales horribly bad
			 */
			if ((!elim)&&(SDO(MASKHIDDEN)))
			{
				err=SetEliminateHidden(S, level);
				if (err>=0)
					elim+=err;
				else
					return err;
			}
			if ((!elim)&&(SDO(MASKINTER)))
			{
				err=EliminateInterBlock(S, level);
				if (err>=0)
					elim+=err;
				else
					return err;
			}
			if ((!elim)&&(SDO(MASK))) 
			{
				err=SetEliminate(S, level);
				if (err>=0)
					elim+=err;
				else
					return err;
			}
		}
	}
	if (Check(S))
		return -1;
	v=S_UK(*S);
	return v;
	
}


inline int NumElim(Sudoku *S, int el)
/* determine a measure for "elimination success" of setting an element to a certain value
/* This makes solving hard sudokus faster and easy sudokus slower
 * I figure as easy sudokus are fast anyway this is an advantage
 * This can dramatically improve filling of empty sudokus
 */
{
	int elim=0;
	int j=0;
	for (j=1;j<=S->LINKS[el][0];j++)
		elim+=EL_P(S->M[S->LINKS[el][j]]&S->M[el]); /* we measure the overlap with linked elements */
	return elim;	
}



inline int Choose(Sudoku *S)
{
	/* chooses next element to try backtracking with.
	 * how the element is chosen has a loarge impact on performance 
	 * with, inparticular, hard sudokus (with easy oned backtracking is 
	 * not or little used). Important is to choose an element with few 
	 * options. This routine extends this by looking at direct 
	 * elemination potential by selecting elements with a large overlap 
	 * with its fellow elements in the various blocks. */
	int minp=S->BS, maxelim=-1, elim, el=-1;
	int i, n;
	
	for (i=0;i<S->N;i++)
	{
		if ((n=EL_P(S->M[i]))>1)
		{
			if (minp>n)
			{
				el=i;			
				maxelim=NumElim(S, i);
				minp=n;
			}
			else if (minp==n)
			{
				elim=NumElim(S, i);
				if (elim>maxelim)
				{
					maxelim=elim;
					el=i;
				}
			}
		}	
	}
	return el;		
}

int BruteForce(Sudoku *S, int *NS, MASKINT ***sol, int maxsol, int limitlevel, int STRAT)
/* Last resort is Backtracking
 * this will find (all) solutions, but no more than maxsol */																		
{
	int i, k, minp=0, el=0, ml, v;
	int *lvl1;
	MASKINT V;
	MASKINT *EL;
	MASKINT *MB;
	minp=S->BS;
	if (S_UK(*S)==0)
		return 0;
	MB=malloc(S->N*sizeof(MASKINT));
	lvl1=malloc(S->N*sizeof(MASKINT));
	
	
	/* choose element */
	el=Choose(S);
	if (el<0)
		return -1;
	/* try all possible values and report each solution */
	/* backup the mask */
	for (i=0;i<S->N;i++)
	{
		lvl1[i]=S->lvl1[i];
		MB[i]=S->M[i];
	}		
	EL=&(S->M[el]);
	V=(*EL);
	
	for (k=0;k<S->BS;k++)
	{
		if (V&VX[k]) /* VX[k] is one of the possibilities */
		{
			(*EL)=VX[k];
			GUESS++;
			if (Eliminate(S,el)>=0) /* we first use cheap elimination to see if there is a reason to continue */
			{
				v=S_UK(*S);
				if (v)
					v=LogicSolve(S, &ml, limitlevel, STRAT);
				if (v==0) /* found a solution */
				{
					(*NS)++;
					(*sol)=realloc((*sol), (*NS)*sizeof(int *));
					(*sol)[(*NS)-1]=malloc(S->N*sizeof(MASKINT));
					for (i=0;i<S->N;i++)
						(*sol)[(*NS)-1][i]=S->M[i];
				}			
				if (v>0) /* still not solved, do it again, recursively */
					v=BruteForce(S, NS, sol, maxsol, limitlevel, STRAT); 
				if ((*NS)>maxsol) /* found more than maxsol solutions, give it up */
				{
					for (i=0;i<S->N;i++)
					{
						S->M[i]=MB[i];
						S->lvl1[i]=lvl1[i];
					}
					break;
				}
			}
			/* restore the mask */
			for (i=0;i<S->N;i++)
			{
				S->M[i]=MB[i];
				S->lvl1[i]=lvl1[i];
			}
		}
	}
	free(MB);
	free(lvl1);
	if ((*NS)==0) /* no solutions found */
		return -1;
	return 0;
}


SSTATE Solve(Sudoku *S, int *maxlevel, MASKINT ***sol, int *ns, int maxsol, int limitlevel, int STRAT)	/* sudoku solver, returns the number of unknowns, 
																					maxlevel indicates the difficulty of the sudoku */
{
	int v, i, vold, strat;
	MASKINT *MB;
	(*ns)=0;
	if (Check(S))
		return INVALID;
	(*maxlevel)=1;	
	GUESS=0;
		
	/* strat is all but brute */
	strat=0;
	strat|=(1<<MASKHIDDEN);
	strat|=(1<<MASKINTER);
	strat|=(1<<MASK);
	if (!(strat&STRAT)) /* only brute force, set limitlevel to 1 */
		limitlevel=1;
		
	for (i=0;i<S->N;i++)
		S->lvl1[i]=0;
		
	if (FastLevel1(S)<0)/* start with eliminating all we can */
		return INVALID; 
		
	v=LogicSolve(S, maxlevel, limitlevel, STRAT);
	
	if (v==0) /* solved */
		return SOLVED;
	
	if (v<0) /* error */
		return INVALID;
	
	/* using brute force */	
	if (SDO(BRUTE))
	{
		SSTATE state=UNSOLVED;
		(*maxlevel)=S->BS;
		(*ns)=0;
		MB=malloc(S->N*sizeof(MASKINT));
		for (i=0;i<S->N;i++)
			MB[i]=S->M[i];
		vold=v;
		v=BruteForce(S, ns, sol, maxsol, limitlevel, STRAT);
		if ((*ns)==0)
		{
			for (i=0;i<S->N;i++)
				S->M[i]=MB[i];
			state=INVALID;			
		}
		if ((*ns)==1) /* one solution */
		{
			for (i=0;i<S->N;i++)
				S->M[i]=(*sol)[0][i];
			free((*sol)[0]);
			state=SOLVED;
		} else if ((*ns)>1) /* more than one solution, reset sudoku */
		{
			v=vold; /* reset number of unknowns */
			for (i=0;i<S->N;i++)
				S->M[i]=MB[i];
			state=MULTISOL;
		}
		free(MB);
		return state;
	}
	return UNSOLVED;
}


int FillEmptySudoku_r(Sudoku *S, int limitlevel)
/* sudoku filler, use logic + backtracking to find 1 solution to an empty sudoku */
{
	int i, j, v, ml, strat, el;
	int k=0;
	MASKINT *MB, M;
	int *lvl1;
	v=S_UK(*S);
	if (v==0)
		return 0;
	fflush(stdout);
		
	MB=malloc(S->N*sizeof(MASKINT));
	lvl1=malloc(S->N*sizeof(MASKINT));
	for (i=0;i<S->N;i++)
	{
		lvl1[i]=S->lvl1[i];
		MB[i]=S->M[i];
	}
	/* strat is all but brute */
	strat=0;
	strat|=(1<<MASKHIDDEN);
	strat|=(1<<MASKINTER);
	strat|=(1<<MASK);		
	/* recursive, backtracking, sudoku filler */	
	/* we go block wise through */

	el=Choose(S);
	if (el<0)
		return -1;
	/* try all values for element el */
	while (k<S->BS)
	{
		if (S->M[el]&VX[k])
		{
			S->M[el]=VX[k];
			M=VX[k];
			v=Eliminate(S,el);
			if (v>=0)
			{
				limitlevel--; 	
				if (limitlevel<1)
					limitlevel=1;
				v=S_UK(*S);
				if (v)
					v=LogicSolve(S, &ml, limitlevel, strat);
			}
			if (v>0)
				v=FillEmptySudoku_r(S, limitlevel); /* call recursively with the next element */
			
			if (v<0) /* conflict */
			{
				limitlevel++;
				if (limitlevel>S->BS)
					limitlevel=S->BS;
					
				for (j=0;j<S->N;j++)
				{
					S->lvl1[j]=lvl1[j];
					S->M[j]=MB[j];
				}
				S->M[el]&=(~M); /* eliminate VX[k] from S->M[el] as it leads to an invalid sudoku */
				v=Eliminate(S,el);
				if (v>=0)
				{
					v=S_UK(*S);
					if (v)
						v=LogicSolve(S, &ml, limitlevel, strat);
				}
				
				if (v<0)
				{
					free(MB);
					free(lvl1);
					return -1; /* permanent conflict, backtrac*/
				}
			} else
			{
				free(MB);
				free(lvl1);
				return 0;
			}
		}
		k++;
	}
	free(MB);
	free(lvl1);
	return -1; /* permanent conflict, tried all options for element el*/
}

void ClearSudoku(Sudoku *S)
{
	MASKINT M=0;
	int i;
	for (i=0;i<S->BS;i++)
		M|=VX[i];
	/* empty sudoku */
	for (i=0;i<S->N;i++)
	{
		S->M[i]=M;
		S->lvl1[i]=0;
	}
}	

int FillEmptySudoku(Sudoku *S)
{
	int r;
	ClearSudoku(S);
	r=FillEmptySudoku_r(S, 1);
	return r;
}

double *ComputeNumVals(Sudoku *S)
{
	double *val, n;
	int i,j, k;
	
	val=calloc(S->BS, sizeof(double));
	
	for (i=0;i<S->NBLK;i++)
	{		
		for (j=0;j<S->BS;j++)
		{
			n=0;
			for (k=0;k<S->NBLK;k++)
				if (S->IBLKS[S->BLKS[i][j]][k])
					n+=1.0; /* count number of blocks this element is part of */
			for (k=0;k<S->BS;k++)
				val[k]+=1.0/(((double)S->BS)*n);
		}
	}
	return val;
}


int CountDoubles(Sudoku *S)
{
	MASKINT M;
	int i,j, D=0;
	for (i=0;i<S->NBLK;i++)
	{
		M=0;
		for (j=0;j<S->BS;j++)
		{
			if (M&S->M[S->BLKS[i][j]])
				D++;
			M|=S->M[S->BLKS[i][j]];
		}
	}
	return D;
}



int * SymbolQuantity(Sudoku *S)
{
	int *res, T=0;
	double *dres, bs;
	int i,j, k;
	double n;
	
	res=malloc(S->BS*sizeof(int));
	dres=calloc(S->BS,sizeof(double));
	bs=(double)S->BS;
	for (i=0;i<S->NBLK;i++)
	{
		for (j=0;j<S->BS;j++)
		{
			n=0;
			for (k=0;k<S->NBLK;k++)
				if (S->IBLKS[S->BLKS[i][j]][k])
					n+=1.0; /* number of blocks this element is a part of */				
			dres[j]+=1.0/(n*bs);
		}
	}
	for (j=0;j<S->BS;j++)
	{
		res[j]=(int)dres[j];
		T+=res[j];
	}
	if (T<S->N)
	{
		int d, dd;
		dd=(S->N-T)/S->BS;
		d=(S->N-T)%S->BS;			
		for (k=0;k<S->BS;k++)
			res[k]+=dd;
		for (k=0;k<d;k++)
			res[k]++;	
	} else if (T>S->N)
	{
		int d, dd;
		dd=(T-S->N)/S->BS;
		d=(T-S->N)%S->BS;	
		for (k=0;k<S->BS;k++)
			res[k]-=dd;
		for (k=0;k<d;k++)
			res[k]--;	
	
	}
	free(dres);
	return res;	
}




int ResolveConflict(Sudoku *S, int num, int *D)
{
	
	MASKINT M, N, T, UK;
	int MatchDouble=0;
	int i, j, k, l, f;
	
	for (i=0;i<S->NBLK;i++)
	{
		M=0;
		N=0;
		j=0;
		while ((!N)&&(j<S->BS))
		{
			N|=(M&S->M[S->BLKS[i][j]]);
			M|=(VX[num]&S->M[S->BLKS[i][j]]);
			j++;
		}
		if (N)
		{
			/* find a missing values in block i */
			UK=0;
			for (k=0;k<S->BS;k++) /* get all options open mask */
				UK|=VX[k];
			M=0;
			for (k=0;k<S->BS;k++)
				M|=S->M[S->BLKS[i][k]];
			T=UK&(~M); /* all missing values in block i */
			
			/* now search a corresponding double */
			for (k=0;k<S->NBLK;k++)
			{
				if (k!=i)
				{
					M=0;
					N=0;
					l=0;
					while ((!N)&&(l<S->BS))
					{
						N|=(M&S->M[S->BLKS[k][l]]);
						M|=(T&S->M[S->BLKS[k][l]]);
						l++;
					}
					f=1 + (S->NBLK-(k+1)); /* factor to make the chance of accepting a change increase as we are further down the loops */
					if (N)
					{
						int p;
						MASKINT DUMMY;
						MatchDouble=1;
						/* double in block l may fix missing in block i */
						/* double num'th value in block i */
						// printf("Swappin M[%d]=%d with M[%d]=%d\n", S->BLKS[i][j-1], EL_V(S->M[S->BLKS[i][j-1]], S->BS), S->BLKS[k][l-1], EL_V(S->M[S->BLKS[k][l-1]], S->BS));
						DUMMY=S->M[S->BLKS[i][j-1]];
						S->M[S->BLKS[i][j-1]]=S->M[S->BLKS[k][l-1]];
						S->M[S->BLKS[k][l-1]]=DUMMY;
						p=CountDoubles(S);
						if (p<(*D))
						{
							/* success, we reduced the number of inconsistensies!*/
							(*D)=p;
							return 1;
						}
						if (rand()<(*D)*RAND_MAX/(f*p)) /* likelyhood of acceptance increases with k (we have to choose something), and decreases if D/p decreases. */
						{
							/* no success, but we accept the change anyway */
							(*D)=p; /* update number of doubles */
							return 1;
						}
						else
						{
							/* we did not reduce the number of inconsistensies!*/
							DUMMY=S->M[S->BLKS[i][j-1]];
							S->M[S->BLKS[i][j-1]]=S->M[S->BLKS[k][l-1]];
							S->M[S->BLKS[k][l-1]]=DUMMY;
						}
							
					}
				}	
			}
			/* no corresponding double, just fix the value then */
			if (!MatchDouble)
			{
				for (k=0;k<S->BS;k++)
					if (VX[k]&T)
					{
						S->M[S->BLKS[i][j-1]]=VX[k];
						return 1;
					}
				fprintf(stderr, "Error: a bug ate the missing value.\n");
				exit(1);
			}
			return 1;
		}
	}
	return 0;
}



int PerfectSquare(int i)
{
    int i2;
    double f;
 
    f=sqrt((double)i);
    i2=(int)f;
 
    if(i2*i2==i)
        return i2;
    else
        return -1;
}

#define MAXITER_NUM 2
#define MAXITER   (S->BS*10000000)
#define MAXNOINCR (S->BS*10000000)
int ResolveConflicts(Sudoku *S)
{
	MASKINT *MB;
	MASKINT M;
	int num, swap=0, s;
	int iter=0, SinceLastImprovement=0, D, D0, k;
	int i1, i2;
	int N, N0;
	time_t curtime;	
	time( &curtime );
	srand( (unsigned  int) curtime );
	
	if ((S_UK(*S)==S->N)&&(S->BS*S->BS==S->N)&&((num=PerfectSquare(S->BS))>0))
	{
		/* assume a standard sudoku structure */
		int i, j, k, l, m;
		
		for (m=0;m<S->N;m++)
		{
			i=m/(num*num*num); 		// block row
			j=(m/(num*num))%num; 	// row in block
			l=m%S->BS;				// column
			k=l/num;				// block col 
			l=l%num; 				// col in block
			S->M[m]=VX[(i+k*num+l+j*num)%S->BS];
		}
	}
	
	if (S_UK(*S)!=0)
	{
		/* fill in empty elements with symbols which are missing */
		int *SQ;
		int i=0;
		SQ=SymbolQuantity(S);
		for (num=0;num<S->N;num++)
		{
			int c;
			c=0;
			do
			{
				c++;
				i++;
				i=i%S->BS;
			} while ((SQ[i]<=0)&&(c<S->BS));
			if (SQ[i]<=0)
			{
				fprintf(stderr, "Error: symbol quantity array is wrong!\n");
				exit(1);
			}			
			S->M[num]=VX[i];
			SQ[i]--;
		}
		free(SQ);	
	}
	D=CountDoubles(S);
	D0=D;
	MB=malloc(S->N*sizeof(MASKINT));
	for (k=0;k<S->N;k++)
		MB[k]=S->M[k];
	N=S->BS/2;	
	printf("%d %.1f %%", iter, 100*((double)(S->N-D))/((double)S->N));
	fflush(stdout);
	while (D&&(SinceLastImprovement<MAXNOINCR))
	{
		i1=-1;
		i2=-1;
		for (num=0;num<S->BS;num++)
		{
			s=1;
			swap=0;
			while((s)&&(swap<MAXITER_NUM))
			{
				s=ResolveConflict(S, num, &D);
				swap+=s;
				if (D==0)
				{
					s=0;
					num=S->BS;
				}
			}
			iter+=swap;
		}
		
		if (N>S->BS/2)
			N=S->BS/2;
		if (D>D0)
		{
			double a;
			a=(double)RAND_MAX*(double)SinceLastImprovement/(double)MAXNOINCR;
			printf("\r%d %3.1f %% %10f", iter, 100*((double)(S->N-D0))/((double)S->N), a/(double)RAND_MAX);
			if (rand()>(int)(0.8*a)) /* do not stray too far from the optimum, certainly in the beginning when it should be easy to improve*/
			{
				/* gravitate to optimum */
				for (k=0;k<S->N;k++)
					S->M[k]=MB[k];
				D=D0;
			}
			else
			{
				/* big disrupt */
				if (rand()<RAND_MAX/10)
				{
					N=S->BS;
					for (k=0;k<N;k++)
					{
						i1=(int) (1.0*(S->N)*rand()/(RAND_MAX+1.0));
						i2=(int) (1.0*(S->N)*rand()/(RAND_MAX+1.0));
						M=S->M[i1];
						S->M[i1]=S->M[i2];
						S->M[i2]=M;
					}
					/*i1=(int) (1.0*(S->N)*rand()/(RAND_MAX+1.0));
					i2=(int) (1.0*(S->N)*rand()/(RAND_MAX+1.0));
					S->M[i1]=S->M[i2];	*/
					D=CountDoubles(S);
				}				
			}
			SinceLastImprovement++;
		}
		else
		{
			for (k=0;k<S->N;k++)
				MB[k]=S->M[k];
			if (D<D0)
			{
				printf("\r%d %.1f %% ", iter, 100*((double)(S->N-D))/((double)S->N));
				fflush(stdout);
				SinceLastImprovement=0;
			}
			D0=D;
		}		
		
		if (D)
		{
			/* small disrupt */
			N=1;
			for (k=0;k<N;k++)
			{
				i1=(int) (1.0*(S->N)*rand()/(RAND_MAX+1.0));
				i2=(int) (1.0*(S->N)*rand()/(RAND_MAX+1.0));
				M=S->M[i1];
				S->M[i1]=S->M[i2];
				S->M[i2]=M;
			}
			D=CountDoubles(S);
		}			
	}
	free(MB);
	printf("\rafter %d iterations %.1f %% consistency is achieved\n", iter, 100*((double)(S->N-D))/((double)S->N));
	return D;
}

