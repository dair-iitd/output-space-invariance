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
#include <string.h>
#include <ctype.h>
#include "sudoku.h"
#include "sudokusolver.h"

#define MAXLINE 5000

void ReadBlockMatrix (Sudoku *S, FILE *f)
{
	char line[MAXLINE+1];
	int NB;
	int *M;
	int lc, k=0, i;
	int *BL, *BLI, Na=10;
	
	M=malloc(S->N*sizeof(int));
	BL=malloc(Na*sizeof(int));
	
	
	
	while(fgets(line, MAXLINE,f)&&(k<S->N))
	{
		lc=0;
		while (line[lc])
		{
			i=0;
			if (isdigit(line[lc]))
			{
				if (k==S->N)
				{
					fprintf(stderr, "Error: too many elements in block matrix definition\n");
					exit(1);
				}
				
				sscanf(line+lc,"%d",&i);
				M[k]=i;
				k++;
				while ((line[lc])&&(isdigit(line[lc])||(line[lc]=='.')))
					lc++;
			} else if (line[lc]=='.')
			{
				if (k==S->N)
				{
					fprintf(stderr, "Error: too many elements in block matrix definition\n");
					exit(1);
				}
				M[k]=-1;
				k++;
				while ((line[lc])&&(isdigit(line[lc])||(line[lc]=='.')))
					lc++;
			} else
				lc++;
		}
		
	}
	if (k<S->N)
	{
		fprintf(stderr, "Error: too few elements in block matrix definition\n");
		exit(1);
	}
	
	NB=0;
	i=-1;
	while (i<S->N-1)
	{
		i++;
		if (M[i]<0)
			continue;
		
		for (k=0;k<NB;k++)
			if (BL[k]==M[i])
				break;
		if (k==NB)
		{
			BL[k]=M[i];
			NB++;
			if (NB==Na)
			{
				Na+=10;
				BL=realloc(BL, Na*sizeof(int));				
			}
		}
	
	}
	BLI=calloc(NB, sizeof(int));
	if (gss_verbose)
		printf("Adding %d custom blocks from Matrix\n", NB);
	S->BLKS=realloc(S->BLKS, (S->NBLK+NB)*sizeof(int *));
	for (i=0;i<NB;i++)
		S->BLKS[S->NBLK+i]=malloc(S->BS*sizeof(int));
		
	for (i=0;i<S->N;i++)
	{
		if (M[i]<0)
			continue;
		for (k=0;k<NB;k++)
			if (BL[k]==M[i])
				break;
		S->BLKS[S->NBLK+k][BLI[k]]=i;
		BLI[k]++;
		if (BLI[k]>S->BS)
		{
			fprintf(stderr, "Error: Block in matrix block definition is larger than the sudoku block size\n");
			exit(1);
		}
	}
	S->NBLK+=NB;
	for (i=0;i<NB;i++)
		if (BLI[i]!=S->BS)
		{
			fprintf(stderr, "Incomplete block in matrix block definition\n");
			exit(1);
		}
	free(BLI);
	free(BL);
	free(M);
	
}

int IsNumberPattern(char *buf, char *pattern, int *incr)
{
	int i=0, r=0;
	char *b;
	char *num;
	
	while (pattern[i])
	{
		if (pattern[i]=='.')
			r++;
		i++;
	}
	if (r!=1)
	{
		fprintf(stderr, "Error: invalid pattern %d\n",r);
		exit(1);
	}
	num=malloc((strlen(buf)+1)*sizeof(char));
	b=buf;
	i=0;
	while ((*pattern)&&(*b))
	{
		if ((*b)!=(*pattern))
			if (!isdigit(*b)||((*pattern)!='.'))
			{				
				free(num);
				return -1;
			}
		if ((isdigit(*b)||(*b=='.'))&&((*pattern)=='.'))
		{
			while ((isdigit(*b)||(*b=='.'))&&((*pattern)=='.'))
			{
				if ((*b)=='.')
					num[i]='0';
				else
					num[i]=*b;
				i++;
				b++;
			}
		}
		else			
			b++;
		pattern++;
	}
	if (*pattern)
	{				
		free(num);
		return -1;
	}
		
	(*incr)=b-buf;
	num[i]='\0';
	r=atoi(num);
	
	free(num);
	return r;
}

void ReadSudokuFormat (Sudoku *S, FILE *f, char *pattern)
{
	char line[MAXLINE+1];
	int go=1;
	int Nla=100;
	int lc,n=0, k, cc=0, i, inc;
	
	
	S->M=malloc(S->N*sizeof(MASKINT));
	S->Npos=malloc((S->N+1)*sizeof(int));
	S->format=malloc(Nla*sizeof(char));
	if (gss_verbose)
		printf("Reading Sudoku Data and Formating\n");
	
	while(fgets(line, MAXLINE,f)&&(go))
	{
		if (strncmp(line, "<endsudoku>", 11)==0)
			go=0;
		else
		{
			lc=0;
			while (line[lc])
			{
				k=0;
				if ((k=IsNumberPattern(line+lc, pattern, &inc))>=0)
				{
					if (n==S->N)
					{
						fprintf(stderr, "Error: too many elements in sudoku format\n");
						exit(1);
					}
					
					if (k>S->BS)
					{
						fprintf(stderr, "Error: digit larger than block size elemnt %d has value %d\n", n, k);
						exit(1);
					}
					if (k==0)
					{
						S->M[n]=0;
						for (i=0;i<S->BS;i++)
							S->M[n]|=VX[i];					
					}
					else
						S->M[n]=VX[k-1];
					S->Npos[n]=cc;
					n++;
					lc+=inc;
				}
				else if (line[lc]== '"')
				{
					lc++;
					while (line[lc]&&(line[lc]!='"'))
					{
						S->format[cc]=line[lc];
						cc++;
						if (cc==Nla)
						{
							Nla+=100;
							S->format=realloc(S->format, Nla*sizeof(char));			
						}
						lc++;
					}
					lc++;
				}
				else
				{
					S->format[cc]=line[lc];
					cc++;
					if (cc==Nla)
					{
						Nla+=100;
						S->format=realloc(S->format, Nla*sizeof(char));			
					}
					lc++;
				}
			}
		}
	}
	S->Npos[n]=-1;
	S->format[cc]='\0';	
	if (n<S->N)
	{
		fprintf(stderr, "Error: too few elements in sudoku format\n");
		exit(1);
	}	
}
typedef enum PARSE {
	COMMENT=0,
	BLOCK, 
	STD_BLOCKS, 
	ROW_BLOCKS, 
	COL_BLOCKS,  
	DOWNDIAG_BLOCK, 
	UPDIAG_BLOCK, 
	BLOCKBLOCKS, 
	MATRIX, 
	SUDOKU, 
	SIZE, 
	BLOCKSIZE,
	PATTERN,
	EMPTYCHAR,
	NONE
}PARSE;

PARSE ParseSudoku(char *line)
{
	char *table[]={
		"#",
		"<block>",
		"<standardblocks>",
		"<rowblocks>",
		"<colblocks>",
		"<downdiagblock>",
		"<updiagblock>",
		"<blockblocks>",
		"<matrix>",
		"<sudoku>",
		"<size>",
		"<blocksize>",
		"<pattern>",
		"<emptychar>",
		NULL
	};
	int i=0, len;
	while (table[i]!=NULL)
	{
		len=strlen(table[i]);
		if (strncmp(table[i], line, len)==0)
			return (PARSE) i;
		i++;
	}
	return NONE;	
}


Sudoku S_Read(char *fn)
{
	FILE *f;
	Sudoku S;
	char pattern[MAXLINE+1];
	char line[MAXLINE+1];
	int i, r=0;
	PARSE P;
	if ((f=fopen(fn, "r"))==NULL)
	{
		fprintf(stderr, "cannot open file %s\n", fn);
		exit(1);
	}
	S.uk='.';
	S.N=0;
	S.BS=0;
	S.NBLK=0;
	S.BLKS=malloc(sizeof(int *));
	pattern[0]='.';
	pattern[1]='\0';
	
	if (gss_verbose)
	{
		printf("* Reading a Sudoku from file: -------------\n");
		printf("file: %s\n", fn);
	}
	while(fgets(line, MAXLINE,f)&&(!r))
	{
		int lc=0;
		P=ParseSudoku(line);
		switch (P)
		{
			case COMMENT:	
				printf("%s", line);
				break;
			case BLOCK:
			{
				int n=0, nn;
				if (gss_verbose)
					printf("Custom Block:");
				if (!S.BS)
				{
					if (gss_verbose)
						printf("No block size given, assuming 9\n");
					S.BS=9;
				}
				S.NBLK++;
				S.BLKS=realloc(S.BLKS, (S.NBLK)*sizeof(int *));
				S.BLKS[S.NBLK-1]=malloc(S.BS*sizeof(int));
				while ((line[lc])&&(n<S.BS))
				{
					nn=0;
					if (sscanf(line+lc,"%d",&nn)==1)
					{
						if (nn<=0)
						{
							fprintf(stderr, "Error: elment index less than 1 in a block definition\n");
							exit(1);
						}
						if (nn>S.N)
						{
							fprintf(stderr, "Error: an elment index exceeds the number of elements in a block definition\n");
							exit(1);
						}
						S.BLKS[S.NBLK-1][n]=nn-1;
						if (gss_verbose)
							printf(" %d",S.BLKS[S.NBLK-1][n]+1);
						n++;
					}
					
					do
					{
						lc++;
					} while (!isspace(line[lc])&&(line[lc]));
				}
				printf("\n");
				break;
			}
			case STD_BLOCKS:
			{
				if (gss_verbose)
					printf("Standard Sudoku\n");
				if (!S.N)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku size for standard blocks\n");
					S.N=81;
				}
				if (!S.BS)
				{
					printf("Assuming standard sudoku block size for standard blocks\n");
					S.BS=9;
				}
				/* standard blocks supported for */
				/* 4x4 		(4 	2x2 squares + 4 rows and 4 columns)
				 * 9x9 		(9 	3x3 squares + 9 rows and 9 columns)
				 * 16x16	(16	4x4 squares + 16 rows and 16 columns)
				 * 25x25	(25	5x5 squares + 25 rows and 25 columns) 
				 * beyond this you need to have 64 bit enabled  
				 * 36x36	(36	6x6 squares + 36 rows and 36 columns)
				 * 49x49	(49 7x7 squares + 49 rows and 49 columns)
				 * 64x64	(64 8x8 squares + 64 rows and 64 columns)
				 * beyond this you need to have 128 bit enabled  
				 * 81x81	(81	9x9 squares + 81 rows and 81 columns)
				 * 100x100	(100 10x10 squares + 100 rows and 100 columns)
				 * 121x121	(121 11x11 squares + 121 rows and 121 columns)
				 * after than you run out of bits! */
				for (i=2;i*i<=(int)(8*sizeof(MASKINT));i++)
				{
					int N, BS;
					BS=i*i;
					N=BS*BS;
					if ((S.N==N)&&(S.BS==BS))
						break;
				}
				
				if (i*i>(int)(8*sizeof(MASKINT)))
				{
					fprintf(stderr, "Error: Standard blocks only works for standard sized sudokus\n");
					exit(1);
				}
				
				
				S.BLKS=realloc(S.BLKS, (S.NBLK+3*S.BS)*sizeof(int *));
				for (i=0;i<3*S.BS;i++)
				{
					if (i<S.BS)
						S.BLKS[S.NBLK+i]=RowBlock(S, i); 
					else if (i<2*S.BS)
						S.BLKS[S.NBLK+i]=ColBlock(S, i-S.BS);
					else
						S.BLKS[S.NBLK+i]=Block(S, i-2*S.BS);
					
				}
				S.NBLK+=3*S.BS;
				break;
			}
			case ROW_BLOCKS:
			{
				int NB;
				if (gss_verbose)
					printf("Standard Rows\n");
				if (!S.N)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku size for standard blocks\n");
					S.N=81;
				}
				if (!S.BS)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku block size for standard blocks\n");
					S.BS=9;
				}
				NB=S.N/S.BS;
				S.BLKS=realloc(S.BLKS, (S.NBLK+NB)*sizeof(int *));
				
				for (i=0;i<NB;i++)
					S.BLKS[S.NBLK+i]=RowBlock(S, i); 
				S.NBLK+=NB;
				break;
			}
			case COL_BLOCKS:
			{
				int NB;
				if (gss_verbose)
					printf("Standard Columns\n");
				if (!S.N)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku size for standard blocks\n");
					S.N=81;
				}
				if (!S.BS)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku block size for standard blocks\n");
					S.BS=9;
				}
				NB=S.N/S.BS;
				S.BLKS=realloc(S.BLKS, (S.NBLK+NB)*sizeof(int *));
				for (i=0;i<NB;i++)
					S.BLKS[S.NBLK+i]=ColBlock(S, i); 
				S.NBLK+=NB;
				break;
			}
			case DOWNDIAG_BLOCK:
			{
				if (gss_verbose)
					printf("Down-Diagonal block\n");
				if (!S.N)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku size for standard blocks\n");
					S.N=81;
				}
				if (!S.BS)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku block size for standard blocks\n");
					S.BS=9;
				}
				S.BLKS=realloc(S.BLKS, (S.NBLK+1)*sizeof(int *));
				S.BLKS[S.NBLK]=DiagBlock(S, 'd'); 
				S.NBLK++;
				break;
			}
			case UPDIAG_BLOCK:
			{
				if (gss_verbose)
					printf("Up-Diagonal block\n");
				if (!S.N)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku size for standard blocks\n");
					S.N=81;
				}
				if (!S.BS)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku block size for standard blocks\n");
					S.BS=9;
				}
				S.BLKS=realloc(S.BLKS, (S.NBLK+1)*sizeof(int *));
				S.BLKS[S.NBLK]=DiagBlock(S, 'u'); 
				S.NBLK++;
				break;
			}
			case BLOCKBLOCKS:
			{
				if (gss_verbose)
					printf("Standard Blocks\n");
				
				if (!S.N)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku size for standard blocks\n");
					S.N=81;
				}
				if (!S.BS)
				{	
					if (gss_verbose)
						printf("Assuming standard sudoku block size for standard blocks\n");
					S.BS=9;
				}
				for (i=2;i*i<=(int)(8*sizeof(MASKINT));i++)
				{
					int N, BS;
					BS=i*i;
					N=BS*BS;
					if ((S.N==N)&&(S.BS==BS))
						break;
				}
				
				if (i*i>(int)(8*sizeof(MASKINT)))
				{
					fprintf(stderr, "Error: Standard blocks only works for standard sized sudokus\n");
					exit(1);
				}
				S.BLKS=realloc(S.BLKS, (S.NBLK+S.BS)*sizeof(int *));
				for (i=0;i<S.BS;i++)
					S.BLKS[S.NBLK+i]=Block(S, i); 
				S.NBLK+=S.BS;
				break;
			}
			case MATRIX:
			{
				if (gss_verbose)
					printf("Block Matrix\n");
				if (!S.N)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku size for standard blocks\n");
					S.N=81;
				}
				if (!S.BS)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku block size for standard blocks\n");
					S.BS=9;
				}
				ReadBlockMatrix (&S, f);
				break;
			}
			case SUDOKU:
			{
				if (!S.N)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku size for standard blocks\n");
					S.N=81;
				}
				if (!S.BS)
				{
					if (gss_verbose)
						printf("Assuming standard sudoku block size for standard blocks\n");
					S.BS=9;
				}
				if (r)
				{
					fprintf(stderr, "Error: one sudoku per file\n");
					exit(1);
				}	
				ReadSudokuFormat (&S, f, pattern);
				r=1;
				break;
			}
			case SIZE:
			{
				if (S.N)
				{
					fprintf(stderr, "Error: sudoku size already defined\n");
					exit(1);
				}
				sscanf(line, "<size>%d", &S.N);
				if (S.N>0)
				{
					if(gss_verbose)
						printf("Setting sudoku size to %d fields\n", S.N);
				}
				else
				{
					fprintf(stderr, "Error: invalid sudoku size %d\n", S.N);
					exit(1);
				}
				break;
			}
			case BLOCKSIZE:
			{
				if (S.BS)
				{
					fprintf(stderr, "Error: block size already defined\n");
					exit(1);
				}
				sscanf(line, "<blocksize>%d", &S.BS);
				if ((S.BS>0)&&(S.BS<=8*sizeof(MASKINT)))
				{
					if (gss_verbose)
						printf("Setting sudoku block size to %d fields\n", S.BS);
				}
				else
				{
					fprintf(stderr, "Error: invalid sudoku block size %d\n", S.BS);
					exit(1);
				}
				break;
			}
			case PATTERN:
				sscanf(line, "<pattern>%s", pattern);
				printf("pattern n= \"%s\"\n", pattern);
				break;
			case EMPTYCHAR:
				sscanf(line, "<emptychar>%c", &S.uk);
				
				break;			
			default:
				break;
		}
	}
	if (!r)
	{
		fprintf(stderr, "Error: did not find a <sudoku> flag\n");
		exit(1);
	}	
	
	if (S.NBLK==0)
	{
		if (gss_verbose)
			printf("No blokcs specified, assuming standard Sudoku\n");
		
		if ((S.N!=81)||(S.BS!=9))
		{
			fprintf(stderr, "Error: Non standard sized sudokus require explicit block specifications\n");
			exit(1);
		}
		S.BLKS=realloc(S.BLKS, (S.NBLK+3*9)*sizeof(int *));
		for (i=0;i<3*9;i++)
		{
			if (i<9)
				S.BLKS[S.NBLK+i]=RowBlock(S, i); 
			else if (i<18)
				S.BLKS[S.NBLK+i]=ColBlock(S, i-9);
			else
				S.BLKS[S.NBLK+i]=Block(S, i-18);
			
		}
		S.NBLK+=3*9;
	}
	fclose(f);
	S.lvl1=malloc(S.N*sizeof(int));
	S.IBLKS=malloc(S.N*sizeof(int *));
	S.LINKS=malloc(S.N*sizeof(int *));
	for (i=0;i<S.N;i++)
	{
		S.IBLKS[i]=malloc(S.NBLK*sizeof(int));
		S.LINKS[i]=malloc((S.BS+1)*sizeof(int));
	}
	PopulateIBlocks(&S);
	if (Check(&S))
		fprintf(stderr, "Warning: detected conflicts in loaded sudoku\n");
	if (gss_verbose)
		printf("-------------------------------------------\n\n");
	return S;

}

FILE * OpenCompactMultiSudokuFile(char *fn)
{
	FILE *f;
	if ((f=fopen(fn, "r"))==NULL)
	{
		fprintf(stderr, "cannot open file %s\n", fn);
		exit(1);
	}
	return f;
}
Sudoku S_InitStdSudoku(int compact)
{
	Sudoku S;
	int i,j,k,l;	
	S.uk='.';	
	S.N=81;
	S.BS=9;
	S.NBLK=0;
	S.BLKS=malloc((3*9)*sizeof(int *));
	S.M=malloc(S.N*sizeof(MASKINT));
	S.Npos=malloc((S.N+1)*sizeof(int));
	S.format=malloc(271*sizeof(char));
	k=0;
	l=0;
	if (!compact)
	{
		S.format=malloc(271*sizeof(char));
		
		for (i=0;i<9;i++)
		{
			if (i%3==0)
			{
				S.format[k++]=' ';
				for (j=0;j<9;j++)
				{
					if (j%3==0)
					{
						S.format[k++]='+';
					}
					else
					{
						S.format[k++]='-';
						S.format[k++]='-';
					}
					S.format[k++]='-';
				}
				S.format[k++]='+';
				S.format[k++]='\n';
			}
			S.format[k++]=' ';
			for (j=0;j<9;j++)
			{
				if (j%3==0)
				{
					S.format[k++]='|';
					S.format[k++]=' ';
				}
					
				S.Npos[l]=k;
				l++;
				S.format[k++]=' ';
			}
			S.format[k++]='|';
			S.format[k++]='\n';
		}
			S.Npos[S.N]=-1;

		S.format[k++]=' ';
		for (j=0;j<9;j++)
		{
			if (j%3==0)
			{
				S.format[k++]='+';
			}
			else
			{
				S.format[k++]='-';
				S.format[k++]='-';
			}
			S.format[k++]='-';
		}
		S.format[k++]='+';
		S.format[k++]='\n';
		S.format[k++]='\0';
		S.format[k++]='\0';
	}
	else
	{
		S.format=malloc(3*sizeof(char));
		
		for (i=0;i<81;i++)
			S.Npos[i]=0;
		S.Npos[S.N]=-1;
		S.format[0]='\n';
		S.format[1]='\0';
		S.format[2]='\0';
	}
	for (i=0;i<3*9;i++)
	{
		if (i<9)
			S.BLKS[i]=RowBlock(S, i); 
		else if (i<18)
			S.BLKS[i]=ColBlock(S, i-9);
		else
			S.BLKS[i]=Block(S, i-18);
		
	}
	S.NBLK=3*9;
	S.lvl1=malloc(S.N*sizeof(int));
	S.IBLKS=malloc(S.N*sizeof(int *));
	S.LINKS=malloc(S.N*sizeof(int *));
	for (i=0;i<S.N;i++)
	{
		S.IBLKS[i]=malloc(S.NBLK*sizeof(int));
		S.LINKS[i]=malloc((S.BS+1)*sizeof(int));
	}
	PopulateIBlocks(&S);
	return S;
}
int S_ReadNextCompactStandard(FILE *f, Sudoku *S)
{
	int i, k;
	char c;
	k=0;		
	while ((feof(f)==0)&&(k<81))
	{
		c=(char)getc(f);
		if (c=='#')
		{
			// read till newline
			while ((feof(f)==0)&&(c!='\n'))
				c=(char)getc(f);				
		}
		if ((c=='0')||(c=='.'))
		{
			S->M[k]=0;
			for (i=0;i<S->BS;i++)
				S->M[k]|=VX[i];
			k++;
		}
		else if (isdigit(c))
		{
			S->M[k]=VX[(int)(c-'1')];
			k++;
		}	
	}
	if (k!=81)
		return 0;
	if (Check(S))
		fprintf(stderr, "Warning: detected conflicts in loaded sudoku\n");
	return 1;
}

void GssOut(Sudoku S, FILE *f, char *pattern) /* export puzzle in gss input format */
{
	int i,j,ds=0;
	fprintf(f,"<size>%d.\n\n", S.N);
	fprintf(f,"<blocksize>%d.\n\n", S.BS);
	fprintf(f,"<emptychar>%c\n\n", S.uk);
	fprintf(f,"<pattern>%s\n\n", pattern);
	for (i=0;i<S.NBLK;i++)
	{
		fprintf(f,"<block> ");
		for (j=0;j<S.BS;j++)
			fprintf(f,"%d ",S.BLKS[i][j]+1);
		fprintf(f,"\n");
	}	
	i=0;
	j=0;
	fprintf(f,"<sudoku>\n");
	i=S.BS;
	while (i/=10)
		ds++;
	while(S.format[i])
	{
		if (S.Npos[j]==i)
		{
			if (S.M==0)
				fprintf(f,"%s",pattern);
			else
			{	
				int k=0, num=0,v;
				while (pattern[k])
				{
					if ((num==0)&&(pattern[k]=='.'))
					{
						v=EL_V(S.M[j], S.BS);
						fprintf(f, "%0*d", ds, v);
						num=1;
					}
					else
						fprintf(f,"%c",pattern[k]);
					k++;
				}
			}
			j++;
		}	
		fprintf(f,"%c",S.format[i]);
		i++;
	}
	fprintf(f,"\n");
}
	
