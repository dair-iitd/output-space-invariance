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
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "sudoku.h"
#include "sudokusolver.h"

int Random(int rmin, int rmax)
{
	return rmin + (int) (1.0*(rmax-rmin+1) * rand()/(RAND_MAX+1.0) );
}


void InitRandom(void)
{
	struct timeval t1;
	gettimeofday(&t1, NULL);
	srand(t1.tv_usec * t1.tv_sec); /* in scripts we may get several times the same seed using srand(time(NULL)) so here we include millisenconds */
	Random(0, 1000);
}


void Shuffle (int N, int *nums)
/* shuffle integer array */
{
	int i;
	int p;
	int dummy;
	for(i=1;i<=N;i++)
	{
		p=Random(0,32767)%i;
		dummy=nums[p];
		nums[p]=nums[i-1];
		nums[i-1]=dummy;
	}
}

int * InterchangableBlocks(Sudoku *S, int b1, int b2)
/* see if blocks i and j are interchangable 
 * if so return a map by which one can change elements in i by elements in j*/
{
	int *res;
	int *B1, *B2;
	int i,j,k;
	B1=S->BLKS[b1];
	B2=S->BLKS[b2];
	res=malloc(S->BS*sizeof(int));
	for (i=0;i<S->BS;i++)
	{
		int go=1;
		/* check element B1[i] */
		res[i]=-1;
		for (j=0;j<S->BS;j++)
		{
			go=1;
			/* search for equivalent */
			k=0;
			while (go&&k<i)
			{
				if (B2[j]==res[k]) /* element B2[j] previously stored as equivalent of some previous B1[i]*/
					go=0;
				k++;
			}
			k=0;			
			while(go&&(k<S->NBLK))
			{
				if ((k!=b1)&&(k!=b2))
					if(S->IBLKS[B1[i]][k]!=S->IBLKS[B2[j]][k])
						go=0;
				k++;						
			}
			if (go) /* element B1[i] is interchangable with B1[j] (i.e. present in the exact same blocks) */
			{
				res[i]=B2[j];
				break;
			}
		}
		if (res[i]<0)
		{
			/* if one element of B1 is not interchangable with any element from B2, the blocks are not interchangable */
			free(res);
			return NULL;
		}			
	}
	/* If we get here all emenets in B1 have some interchangable element in B2[j] 
	 * the result struct maps the elemnt S->BLKS[b1][i] <-> res[i] */
	return res;
}

#define Pres 1000 /* resolution for the probability, i.e., Pres=1000 lets you define probabilities with a resolution of 1 promille */
int Doit(double Pdoit)
{
	if (Pdoit<=0)
		return 0;
	
	if (Pdoit>=1)
		return 1;
		
	return (Random(0, Pres)<round(Pres*Pdoit));
	
}
#undef Pres

int InterchangBlocks(Sudoku *S, double Pdoit)/* count number of possible interchanges, and do it with a probablity of 1 in Pdoit and 0 is Pdoit==0 */
{
	int i,j, k, n=0;
	int *map;
	for (i=0;i<S->NBLK;i++)
		for (j=i+1;j<S->NBLK;j++)
		{
			map=InterchangableBlocks(S, i, j);
			if (map) /* blocks may be interchanged */
			{
				n++;
				if (Doit(Pdoit))/* we do it */
				{
					MASKINT dummy;
					for (k=0;k<S->BS;k++)
					{
						dummy=S->M[S->BLKS[i][k]];
						S->M[S->BLKS[i][k]]=S->M[map[k]];
						S->M[map[k]]=dummy;
					}
				}
				free(map); 
			}
			
		}
	return n;
}

int Rotate180(Sudoku *S, double Pdoit) /* test if rotatable by 180 degrees. If doit=TRUE it will rotate if possible, otherwise only test */
{
	MASKINT dummy;
	int r;
	int i=0;
	while (S->N-1-i>i) /* reverse array */
	{
		dummy=S->M[i];
		S->M[i]=S->M[S->N-1-i];
		S->M[S->N-1-i]=dummy;
		i++;
	}
	if (((r=Check(S))==1)||(!Doit(Pdoit)))
	{/* undo change */
		i=0;
		while (S->N-1-i>i)
		{
			dummy=S->M[i];
			S->M[i]=S->M[S->N-1-i];
			S->M[S->N-1-i]=dummy;
			i++;
		}
	}
	return !r;	
}


int Rotate90Index(int i, int N)
{
	return (i%N)*N+N-(i/N)-1;	
}

static int intsqsrt(int val) {
    int a,b,y;
    a = 1;
    b = val;
    do
	{
		y=(a+b)/2; 
		if (y*y>val)
			b=y;
		else
			a=y;
	} while(b-a>1);
	if (val-a*a<b*b-val)
		return a;
    return b;
}

int Rotate90(Sudoku *S, double Pdoit) /* test if rotatable by 90 degrees. If doit=TRUE it will rotate if possible, otherwise only test */
{
	MASKINT *dummy, *M;
	int i, f;
	int r;
	f=intsqsrt(S->N);
	
	if (f*f!=S->N)
		return 0; /* sudoku must be square to still work after rotating 90 degrees */	
	M=malloc(S->N*sizeof(MASKINT));
	for (i=0;i<S->N;i++)
		M[i]=S->M[Rotate90Index(i, f)];
	dummy=S->M;
	S->M=M;
	if (((r=Check(S))==1)||(!Doit(Pdoit)))
	{
		free(S->M);
		S->M=dummy;
	}
	else
		free(dummy);
	return !r;	
}

int HmirrorIndex(int i, int N)
{
	return i+N-2*(i%N)-1;
}
int VmirrorIndex(int i, int N, int M)
{
	return (M-i/N-1)*N+i%N;
}
int Mirror(Sudoku *S, double Pdoit, char dir) /* test if rotatable by 90 degrees. If doit=TRUE it will rotate if possible, otherwise only test */
{
	MASKINT *dummy, *M;
	int i, f;
	int r, R=0;	
	if ((dir!='v')&&(dir!='h'))
	{
		fprintf(stderr, "Error: direction should be either \'h\' or \'v\', not \'%c\'\n",dir);
		return 0;
	}
	M=malloc(S->N*sizeof(MASKINT));
	for (f=2;f<S->N-1;f++)
	{
		if (S->N%f==0)
		{
			for (i=0;i<S->N;i++)
				if (dir=='h')
					M[i]=S->M[HmirrorIndex(i, f)];
				else
					M[i]=S->M[VmirrorIndex(i, f, S->N/f)];
			dummy=S->M;
			S->M=M;
			if (((r=Check(S))==1)||(!Doit(Pdoit)))
			{
				M=S->M;
				S->M=dummy;
			}
			else
				M=dummy;
			if (!r)
				R++;
		}
	}
	free(M);
	return R;
}

void Analyze(Sudoku *S)
{
	int i,j, k, ni=0;
	int *map;
	if (S_UK(*S)!=0)
	{
		fprintf(stderr, "Error: Analyze should be called on a solved sudoku\n");
		exit(1);
	}
	
	printf("Sudoku Structure Analysis:\n");
	for (i=0;i<S->NBLK;i++)
		for (j=i+1;j<S->NBLK;j++)
		{
			map=InterchangableBlocks(S, i, j);
			if (map)
			{
				printf("Block %d and %d are interchangable\n", i, j);
				ni++;
				for (k=0;k<S->BS;k++)
					printf("%d <-> %d\n", S->BLKS[i][k], map[k]);
				printf("\n");
				free(map); 
			}
			
		}
	if (ni)		
		printf("Found %d interchangable blocks\n", ni);
	else
		printf("Found no interchangable blocks\n");
	
	if (Rotate180(S, 0))
		printf(  "Sudoku is rotatable by 180 degrees\n");
	else
		printf(  "Sudoku is not rotatable by 180 degrees\n");
	if (Rotate90(S, 0))
		printf(  "Sudoku is rotatable by 90 degrees\n");
	else
		printf(  "Sudoku is not rotatable by 90 degrees\n");
	k=Mirror(S, 0, 'h');
	if (k)
		printf(  "Found %d ways to mirror the sudoku horizontally\n", k);
	else
		printf(  "Could not mirror the sudoku horizontally\n");
	k=Mirror(S, 0, 'v');
	if (k)
		printf(  "Found %d ways to mirror the sudoku vertically\n", k);
	else
		printf(  "Could not mirror the sudoku vertically\n");
	
	
	
}
void ScrambleSudoku(Sudoku *S, double Pdoit)
/* transforms sudoku (interchange blocks, mirror, rotate, ... */
/* some of these operations may be redundant */
{
	int r=0;
	r+=Rotate180(S, Pdoit);
	r+=Rotate90(S, Pdoit);
	r+=Mirror(S, Pdoit, 'v');
	r+=Mirror(S, Pdoit, 'h');
	r+=InterchangBlocks(S, Pdoit);
	if (gss_verbose)
		printf("Found %d scrambling operations\nEach applied with a probablity of %f\n", r, Pdoit);
}

void ShuffleSudoku(Sudoku *S)
/* randomizes the symbols */
/* other randomizations are herd to generalize to different sudoku topologies */
{
	int i;
	int *Nums;
	if (S_UK(*S)!=0)
	{
		fprintf(stderr, "Error: Randomizing sudoku's requires the sudoku to be solved first\n");
		exit(1);
	}
	Nums=malloc(S->BS*sizeof(int));
	
	for(i=0;i<S->BS;i++)
		Nums[i]=i;
	
	Shuffle(S->BS,Nums);
	for(i=0;i<S->N;i++)
		S->M[i]=VX[Nums[EL_V(S->M[i], S->BS)-1]];
	free(Nums);
}
void Randomize(Sudoku *S)
{
	if (gss_verbose)
		printf("Randomizing the sudoku\n");
	InitRandom();
	/* randomize */
	ShuffleSudoku(S);
	ScrambleSudoku(S, 0.5);
}

int GenerateSudoku_sparse(Sudoku *S, int maxlevel, int STRAT)
{
	/* input a solved sudoku, output an unsolved sudoku */
	int i,j, level=0, ml=0;
	int *Nums;
	MASKINT M;
	MASKINT UK;
	MASKINT *MB;
	
	if (maxlevel<S->BS) /* turn off brute force, it serves no purpose here */
	{
		int strat=0;
		strat|=(STRAT&(1<<MASKHIDDEN));
		strat|=(STRAT&(1<<MASKINTER));
		strat|=(STRAT&(1<<MASK));
		STRAT=strat;
	}
	if (maxlevel>1) /* if the level is larger than one at least one high level strategies must be selected */
	{
		if (!((STRAT&(1<<MASKHIDDEN))||(STRAT&(1<<MASKINTER))||(STRAT&(1<<MASK))))
			STRAT|=(1<<MASK);
	}
	
	
	Nums=malloc(S->N*sizeof(int));
	for (i=0;i<S->N;i++)
		Nums[i]=i;
		
	UK=0;
	for (i=0;i<S->BS;i++)
		UK|=VX[i];
	Shuffle(S->N,Nums);

	MB=malloc(S->N*sizeof(MASKINT));
	for(j=0;j<S->N;j++)
		MB[j]=S->M[j];
	
	for(j=0;j<S->N;j++)
	{
		M=MB[Nums[j]];
		MB[Nums[j]]=UK;
		
		for (i=0;i<S->N;i++)
			S->M[i]=MB[i];

		switch(ProcessSudoku(*S, 0, 1, maxlevel, STRAT, &level))
		{
			case SOLVED:
				for (i=0;i<S->N;i++)
					S->M[i]=MB[i];	
				if (gss_verbose>1)
				{		
					printf("New sudoku with level %d and %d unknown:\n", level, S_UK(*S));		
					S_Print(*S, stdout);
				}
				ml=level;
				if (level<=maxlevel)
					break;
			case UNSOLVED:	
			case MULTISOL:
			case INVALID:
			default:
				MB[Nums[j]]=M;			
		}
	}
	for (i=0;i<S->N;i++)
		S->M[i]=MB[i];
	free(Nums);
	free(MB);
	return ml;
}


int GenerateSudoku(Sudoku *S, int maxlevel, int STRAT)
{
	/* input a solved sudoku, output an unsolved sudoku */
	int i,j, k, level=0, ml=0;
	int *Nums;
	MASKINT M;
	MASKINT UK;
	MASKINT *MB;
	
	if (maxlevel<S->BS) /* turn off brute force, it serves no purpose here */
	{
		int strat=0;
		strat|=(STRAT&(1<<MASKHIDDEN));
		strat|=(STRAT&(1<<MASKINTER));
		strat|=(STRAT&(1<<MASK));
		STRAT=strat;
	}
	if (maxlevel>1) /* if the level is larger than one at least one high level strategies must be selected */
	{
		if (!((STRAT&(1<<MASKHIDDEN))||(STRAT&(1<<MASKINTER))||(STRAT&(1<<MASK))))
			STRAT|=(1<<MASK);
	}
	
	
	Nums=malloc(S->N*sizeof(int));
	for (i=0;i<S->N;i++)
		Nums[i]=i;
		
	UK=0;
	for (i=0;i<S->BS;i++)
		UK|=VX[i];
	Shuffle(S->N,Nums);

	MB=malloc(S->N*sizeof(MASKINT));
	for(j=0;j<S->N;j++)
		MB[j]=S->M[j];
	k=0;
	/* first one run at level one to eliminate as many elements fast */
	for(j=0;j<S->N;j++)
	{
		M=MB[Nums[j]];
		MB[Nums[j]]=UK;
		
		for (i=0;i<S->N;i++)
			S->M[i]=MB[i];

		switch(ProcessSudoku(*S,  0, 1, 1, STRAT, &level))
		{
			case SOLVED:
				/*new Nums will not have this value, it is already eliminated */
				for (i=0;i<S->N;i++)
					S->M[i]=MB[i];	
				if (gss_verbose>1)
					printf("%f %% complete %d unknown:\n", (double)j/(double)S->N, S_UK(*S));		
				ml=level;
				if (level<=1)
					break;
			case UNSOLVED:
				/* this number is a candidate for higher level sudokus */
				Nums[k++]=Nums[j];
				MB[Nums[j]]=M;	
				break;
			case MULTISOL:
			case INVALID:
			default:
				MB[Nums[j]]=M;			
		}
	}
	if (gss_verbose>1)
	{		
		printf("New sudoku with level 1  and %d unknown:\n", S_UK(*S));		
		S_Print(*S, stdout);
	}
	k=S->N;
	if (maxlevel>1)
	{
		for(j=0;j<k;j++)
		{
			M=MB[Nums[j]];
			MB[Nums[j]]=UK;
			
			for (i=0;i<S->N;i++)
				S->M[i]=MB[i];

			switch(ProcessSudoku(*S, 0, 1, maxlevel, STRAT, &level))
			{
				case SOLVED:
					for (i=0;i<S->N;i++)
						S->M[i]=MB[i];	
					if (gss_verbose>1)
					{		
						printf("New sudoku with level %d and %d unknown:\n", level, S_UK(*S));		
						S_Print(*S, stdout);
					}
					ml=level;
					if (ml==maxlevel)
						j=k;
					if (level<=maxlevel)
						break;
				case UNSOLVED:	
				case MULTISOL:
				case INVALID:
				default:
					MB[Nums[j]]=M;			
			}
		}
	}
	for (i=0;i<S->N;i++)
		S->M[i]=MB[i];
	free(Nums);
	free(MB);
	return ml;
}
