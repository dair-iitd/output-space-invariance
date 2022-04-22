/*
    JigSawMRF a random field generatopr to creater jigsaw sudokus
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
#include <string.h>
/* A Jigsaw sudoku pattern generator, creates connected "houses" within 
 * a undirected graph, where each house has the same number of 
 * elements. The algirithm treats the graph as a Markov Random Field 
 * (does it, is this still a Markov Random Field??) and computes an 
 * "energy" for each cell which depends on its surroundings. At 
 * initialization the graphs cells are randomly assigned to houses. 
 * Thus the houses are not "connected" regions in the graph. After 
 * initialization an optimization algorithm separates the houses 
 */
typedef struct cell {
	struct cell **Adj;
	int id;
	double *r;
	int Nadj;
	int E;
	int phase;
} cell;

typedef struct Field {
	cell *F;
	int NF;
	int Np;
	int Ndim;
} Field;

int Random(int rmin, int rmax)
{
	return rmin + (int) (1.0*(rmax-rmin+1) * rand()/(RAND_MAX+1.0) );
}

void FreeCell(cell *C)
{
	free(C->Adj);
	C->Adj=NULL;
	C->Nadj=0;
	free(C->r);
}
void FreeField(Field *F)
{
	int i;
	for(i=0;i<F->NF;i++)
		FreeCell(F->F+i);
	free(F->F);
	F->F=NULL;
	F->NF=0;
	F->Ndim=0;
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

#define BLOCK 4
cell **Same_rec(cell **Same, int *Ns, cell *C)
{
	int i, p;
	p=C->phase;
	C->phase=-1; // flood fill with -1 to prevent loops
	for (i=0;i<C->Nadj;i++)
	{
		if (C->Adj[i]->phase==p)
		{
			Same[*Ns]=C->Adj[i]; // add cell to Same array
			(*Ns)++;
			if ((*Ns)%BLOCK==0)
				Same=realloc(Same, ((*Ns)+BLOCK)*sizeof(cell *));	
			Same=Same_rec(Same,Ns,C->Adj[i]); // recurse
		}
	}
	return Same;
}
cell **Neighbourhood(cell *C, int *Ns)
{
	cell **Same;
	int i;
	int p;
	Same=malloc(BLOCK*sizeof(cell *));
	Same[0]=C;
	p=C->phase;
	(*Ns)=1;
	Same=Same_rec(Same, Ns, C);
	for (i=0;i<(*Ns);i++)
		Same[i]->phase=p;
	return Same;
}
int Energy(Field *F, int *minE)
{
	int i, j;
	int Ns;
	int E=0, Emin=F->NF;
	cell **s;
	// reset
	for (i=0;i<F->NF;i++)
		F->F[i].E=-1;
		
	for (i=0;i<F->NF;i++)
	{
		if (F->F[i].E<0)
		{
			s=Neighbourhood(F->F+i, &Ns);
			for (j=0;j<Ns;j++)
				s[j]->E=Ns;
			free(s);
		}
		E=E+F->F[i].E;
		if (F->F[i].E<Emin)
			Emin=F->F[i].E;
	}
	if (minE)
		(*minE)=Emin;
	return E;
}

Field RectField(int n, int m, int Np)
{
	Field F;
	int i,j;
	int Nadj;
	int *p, np;
	
	
	if ((n*m)%Np)
	{
		fprintf(stderr,"Error: number of cells must be divisibly by the number of phases %d %d %d %d\n", n, m, Np, (n*m)%Np);
		exit(1);
	}
	F.NF=n*m;
	F.Ndim=2;
	F.Np=Np;
	F.F=malloc(F.NF*sizeof(cell));
	
	np=n*m/Np;
	p=malloc(F.NF*sizeof(int));
	j=0;
	for (i=0;i<F.NF;i++)
	{
		p[i]=j;
		if (i%np==np-1)
			j++;
	}
	Shuffle (F.NF, p);
	for (i=0;i<F.NF;i++)
	{
		Nadj=0;
		int y,x;
		F.F[i].r=malloc(F.Ndim*sizeof(double));
		F.F[i].Adj=malloc(4*sizeof(cell *));
		y=i%n;
		x=i/n;
		F.F[i].r[0]=(double)x+0.5;
		F.F[i].r[1]=(double)y+0.5;
		if ((y-1)>0)
		{
			F.F[i].Adj[Nadj]=F.F+i-1;
			Nadj++;
		}
		if ((y+1)<n)
		{
			F.F[i].Adj[Nadj]=F.F+i+1;
			Nadj++;
		}
		if ((x-1)>0)
		{
			F.F[i].Adj[Nadj]=F.F+i-n;
			Nadj++;
		}
		if ((x+1)<m)
		{
			F.F[i].Adj[Nadj]=F.F+i+n;
			Nadj++;
		}
		F.F[i].Nadj=Nadj;
		F.F[i].phase=p[i];
		F.F[i].id=i;
	}
	free(p);
	return F;
}


int * GetCell(FILE *f, cell *C, int *Ndim)
{
	int *linkids;
	char *word;
	int Na=10;
	int i=0;
	char ch='a';
	int R=1;
	
	C->Nadj=0;
	C->Adj=NULL;
	C->id=-1;
	C->r=NULL;	
		
	linkids=malloc(Na*sizeof(int));
	if(fscanf(f,"%[#]", &ch)==1) // skip header
	{
		while ((feof(f)==0)&&(fscanf(f,"%c", &ch)))
		{
			if (ch=='\n')
				break;
		}	
		return NULL;
	}
	
	while ((feof(f)==0)&&(ch!='\n'))
	{
		/* get and allocate a word */
		if (fscanf(f,"%ms", &word)==1)
		{
			/* parse word */
			if (R)
			{
				int adim=4, ndim=0;
				int j,k;
				char c;
				if (*word !='(')
				{
					fprintf(stderr,"Error: cannor parse cell coordinate from %s\n", word);
					free(word);
					exit(1);
				}
				C->r=malloc(adim*sizeof(double));
				j=1;
				// parse position
				while (word[j])
				{
					k=j;
					while((word[j])&&(word[j]!=',')&&(word[j]!=')'))
						j++;
					c=word[j];
					word[j]='\0';
					C->r[ndim++]=atof(word+k);
					if (ndim==adim)
					{
						adim+=3;
						C->r=realloc(C->r,adim*sizeof(double));
					}
					if (c)
						j++;
					else
					{
						fprintf(stderr,"Error: ill formatted coordinate, missing a closing bracket %s\n", word);
						free(word);
						exit(1);
					}				
				}
				R=0;
				
				(*Ndim)=ndim;
			}
			else
			{
				linkids[i]=atoi(word);
				i++;
				if (i==Na)
				{
					Na+=10;
					linkids=realloc(linkids,Na*sizeof(int));
				}
			}
			free(word);
		}
		// clear whitespace but no endlines
		while ((feof(f)==0)&&(fscanf(f,"%[ \t]", &ch)==1));
		/* check for an endline */
		fscanf(f,"%[\n]",&ch);
	}
	C->Nadj=i;
	C->id=0;
	return linkids;
}


cell * ReadCustomGraph(char *fn, int *NF, int *Ndim)
{	
	int i,j;
	FILE *f;
	cell *Cs;
	int ndim, pndim=-1;
	int Na=10, Nc=0, **LT;
	if ((f=fopen(fn,"r"))==NULL)
	{
		fprintf(stderr,"Error: cannot open %s for reading\n", fn);
		exit(1);
	}
	Cs=malloc(Na*sizeof(cell));
	LT=malloc(Na*sizeof(int **));
	while (feof(f)==0)
	{
		LT[Nc]=GetCell(f, Cs+Nc, &ndim);
		if (Cs[Nc].id==0)
		{
			Cs[Nc].id=Nc;
			if (ndim<=0)
			{
				fprintf(stderr,"Error: cannot have %d dimensions\n", ndim);
				exit(1);
			}
			if (pndim<0)
				pndim=ndim;
			else if (ndim!=pndim)
			{
				fprintf(stderr,"Error: mismatch in number of dimensions\n");
				exit(1);
			}
			if (Cs[Nc].Nadj==0)
			{
				fprintf(stderr,"Error: got a cell without links\n");
				exit(1);
			}		
			Nc++;
		}
		if (Nc==Na)
		{
			Na+=10;
			Cs=realloc(Cs, Na*sizeof(cell));
			LT=realloc(LT, Na*sizeof(int **));			
		}
	}
	fclose(f);
	(*NF)=Nc;
	(*Ndim)=ndim;
	for (i=0;i<Nc;i++)
	{
		Cs[i].Adj=malloc(Cs[i].Nadj*sizeof(cell *));
		for (j=0;j<Cs[i].Nadj;j++)
		{
			if ((LT[i][j]<1)||(LT[i][j]>Nc))
			{
				fprintf(stderr,"Error: cell link is out of range\n");
				exit(1);
			}		
			Cs[i].Adj[j]=Cs+LT[i][j]-1;
		}
		free(LT[i]);
	}
	free(LT);
	return Cs;
}

Field CustomField(char *fn, int Np)
{
	Field F;
	int i,j;
	int *p, np;
	
	F.F=ReadCustomGraph(fn, &F.NF, &F.Ndim);
	if ((F.NF)%Np)
	{
		fprintf(stderr,"Error: number of cells must be divisibly by the number of phases\n");
		exit(1);
	}
	F.Np=Np;
	np=F.NF/Np;
	p=malloc(F.NF*sizeof(int));
	j=0;
	for (i=0;i<F.NF;i++)
	{
		p[i]=j;
		if (i%np==np-1)
			j++;
	}
	Shuffle (F.NF, p);
	for (i=0;i<F.NF;i++)
		F.F[i].phase=p[i];
	free(p);
	
	return F;
}


int AnnealStep(Field *F, int *Emin)
{
	int el1, el2, p;
	int Eo, En, i, j, emin0;
	int *candidates, Nc;
	
	
	
	Eo=Energy(F, &emin0);
	candidates=malloc(F->NF*sizeof(int));
	
	if (Random(0, 100000)>70000)// 30% probability to take one of the least connected elements seems to be good
	{		
		Nc=0;		
		for (i=1;i<F->NF;i++)
			if (F->F[i].E==emin0)
			{
				candidates[Nc]=i;
				Nc++;
			}
		switch(Nc)
		{
			case 0:
				el1=Random(0, F->NF-1);
				break;
			case 1:
				el1=candidates[0];
				break;
			default:
				el1=candidates[Random(0, Nc-1)];
		}		
	}
	else
		el1=Random(0, F->NF-1);
	p=F->F[el1].phase;
	
	Nc=0;
	En=F->NF;
	for (i=1;i<F->NF;i++)
	{
		for (j=0;j<F->F[i].Nadj;j++)
			if (F->F[i].Adj[j]->phase==p)
			{
				if (En>F->F[i].Adj[j]->E)
					En=F->F[i].Adj[j]->E;
				if (i!=el1)
				{
					candidates[Nc]=i;
					Nc++;
					break;
				}
			}
	}
	
	// pick random element from candidates
	el2=candidates[Random(0, Nc-1)];
	free(candidates);
	
	// swap phase;
	F->F[el1].phase=F->F[el2].phase;
	F->F[el2].phase=p;
	En=Energy(F, Emin);	
	if(Eo-En>0)
	{
		p=F->F[el1].phase;
		F->F[el1].phase=F->F[el2].phase;
		F->F[el2].phase=p;
		if (Emin)
			(*Emin)=emin0;
		return Eo;
	}
	return En;		
}

double * BorderPoints(Field F, int *N)
{
	int i, j, k, Na=10, nb=0;
	double *r;
	r=malloc(F.Ndim*Na*sizeof(double));
	
	for (i=0;i<F.NF;i++)
	{
		for (j=0;j<F.F[i].Nadj;j++)
			if ((F.F[i].phase!=F.F[i].Adj[j]->phase)&&(F.F[i].Adj[j]->id>i))
			{
				for (k=0;k<F.Ndim;k++)
					r[nb*F.Ndim+k]=(F.F[i].r[k]+F.F[i].Adj[j]->r[k])/2;
				nb++;
				if (nb==Na)
				{
					Na+=10;
					r=realloc(r,F.Ndim*Na*sizeof(double));
				}				
			}
	}
	(*N)=nb;
	return r;
}

void PrintField(char *fn, Field F)
{
	int i,k;
	FILE *f;
	if ((f=fopen(fn,"w"))==NULL)
	{
		fprintf(stderr,"Error: cannot open %s for writing\n", fn);
		exit(1);
	}
	for (i=0;i<F.NF;i++)
	{
		for (k=0;k<F.Ndim;k++)
			fprintf(f,"%e ",F.F[i].r[k]);
		fprintf(f,"%d %d\n", F.F[i].phase, F.F[i].E);
	}
	fclose(f);
}
void PrintPhaseEl(char *fn, Field F, int off)
{
	int i,j;
	int *el;
	int *np;
	int m;
	FILE *f;
	if ((f=fopen(fn,"w"))==NULL)
	{
		fprintf(stderr,"Error: cannot open %s for writing\n", fn);
		exit(1);
	}
	el=malloc(F.NF*sizeof(int));	
	np=calloc(F.Np,sizeof(int));
	m=F.NF/F.Np;
	for (i=0;i<F.NF;i++)
	{
		el[F.F[i].phase+np[F.F[i].phase]*F.Np]=i+off;
		np[F.F[i].phase]++;
	}
	
	for (i=0;i<F.Np;i++)
	{
		for (j=0;j<m;j++)
			fprintf(f,"%d ", el[i+j*F.Np]);
		fprintf(f,"\n");
	}
	free(el);
	free(np);
	fclose(f);
}
void PrintBorderPoints(char *fn, Field F)
{
	int i, Nb, k;
	double *r;
	FILE *f;
	r=BorderPoints(F, &Nb);
	if ((f=fopen(fn,"w"))==NULL)
	{
		fprintf(stderr,"Error: cannot open %s for writingßn", fn);
		exit(1);
	}
	for (i=0;i<Nb;i++)
	{
		for (k=0;k<F.Ndim;k++)
			fprintf(f,"%e\t", r[i*F.Ndim+k]);
		fprintf(f,"\n");
	}
	
	fclose(f);
	free(r);
}


typedef enum {RECT,CUSTOM,PHASES,OUTP, OUTB, OUTPE, VERBOSE, OFFSET, NONE} PRSK;
typedef struct {
	PRSK K;
	char *key;
	int l;
} Keyword;
Keyword KeyTable[] = {
	{RECT,   "--rect"    , 6},
	{RECT,   "-r"        , 2},
	{CUSTOM, "--custom"  , 8},
	{CUSTOM, "-c"        , 2},
	{PHASES, "--phases"  , 8},
	{PHASES, "-p"        , 2},
	{OUTP,   "--outp"    , 6},
	{OUTP,   "-P"        , 2},
	{OUTPE,  "--outpe"   , 7},
	{OUTPE,  "-e"        , 2},
	{OUTB,   "--outb"    , 6},
	{OUTB,   "-B"        , 2},
	{VERBOSE,"--verbose" , 9},
	{VERBOSE,"-v"        , 2},
	{OFFSET, "--offset"  , 8},
	{OFFSET, "-o"        , 2},
	{NONE, NULL, 0}
};
PRSK LookupKey(char *w)
{
	int l,i=0;
	l=strlen(w);
	while (KeyTable[i].l)
	{
		if (l==KeyTable[i].l)
			if (strncmp(w,KeyTable[i].key,KeyTable[i].l)==0)
				return KeyTable[i].K;
		i++;
	}
	return NONE;
}
				

int main(int argc, char **argv)
{
	Field F;
	PRSK K;
	int V=0;
	char *outpe=NULL;
	char *outp=NULL;
	char *outb=NULL;
	char *custin=NULL;
	int i, k, E, Emin;
	int N=0, M=0, NP=0;
	int off=0;
	InitRandom();
	printf("JigSawMRF Copyright (C) 2019  B. Pieters\n");
    printf("This program comes with ABSOLUTELY NO WARRANTY.\n");
    printf("This is free software, and you are welcome to redistribute it\n");
    printf("under the conditions of the GPL v3.\n");
	k=1;
	while (k<argc)
	{
		K=LookupKey(argv[k]);
		switch (K)
		{
			case RECT:
				if (k+2>=argc)
				{
					fprintf(stderr, "Error. too few arguments to --rect\n");
					exit(1);
				}
				N=atoi(argv[k+1]);
				M=atoi(argv[k+2]);
				k+=2;
				
				if (N<=0)
				{
					fprintf(stderr, "Error. invalid number of rows\n");
					exit(1);
				}		
				if (M<=0)
				{
					fprintf(stderr, "Error. invalid number of columns\n");
					exit(1);
				}	
				break;
			case CUSTOM:
				if (k+1>=argc)
				{
					fprintf(stderr, "Error. too few arguments to --custom\n");
					exit(1);
				}
				custin=argv[k+1];
				k+=1;
				break;
			case PHASES:
				if (k+1>=argc)
				{
					fprintf(stderr, "Error. too few arguments to --phases\n");
					exit(1);
				}
				NP=atoi(argv[k+1]);
				k+=1;
				if (NP<=0)
				{
					fprintf(stderr, "Error. invalid number of phases\n");
					exit(1);
				}		
				break;
			case OUTP:
				if (k+1>=argc)
				{
					fprintf(stderr, "Error. too few arguments to --outp\n");
					exit(1);
				}
				outp=argv[k+1];
				k+=1;
				break;
			case OUTPE:
				if (k+1>=argc)
				{
					fprintf(stderr, "Error. too few arguments to --outpe\n");
					exit(1);
				}
				outpe=argv[k+1];
				k+=1;
				break;
			case OUTB:
				if (k+1>=argc)
				{
					fprintf(stderr, "Error. too few arguments to --outb\n");
					exit(1);
				}
				outb=argv[k+1];
				k+=1;
				break;
			case VERBOSE:
				if (k+1>=argc)
				{
					fprintf(stderr, "Error. too few arguments to --verbose\n");
					exit(1);
				}
				V=atoi(argv[k+1]);
				k+=1;
				break;
			case OFFSET:
				if (k+1>=argc)
				{
					fprintf(stderr, "Error. too few arguments to --verbose\n");
					exit(1);
				}
				off=atoi(argv[k+1]);
				k+=1;
				break;
			default:
				fprintf(stderr, "Error. unknown option %s\n",argv[k]);
				exit(1);
		} 
		k++;
	}
	if (!custin)
	{
		if ((N<=0)||(M<=0)||(NP<=0))
		{
			fprintf(stderr, "Error. missing or invalid arguments\n");
			exit(1);
		}	
		F=RectField(N, M, NP);
	}
	else
		F=CustomField(custin, NP);
	i=0;
	while(1)
	{
		E=AnnealStep(&F, &Emin);	
		if (V>0)
			if (i%10000==0)
				fprintf(stderr, "\r%d %.1f%% %d", i, 100.0*((double)E)/((double)(F.NF*F.NF)/((double)NP)), Emin);
		i++;
		if (Emin==(F.NF)/NP)
			break;
	}
	fprintf(stderr, "\nNumber of iterations %d\n", i);
	
	Energy(&F, NULL);
	if (outp)
		PrintField(outp,F);
	if (outpe)
		PrintPhaseEl(outpe, F, off+1);
	if (outb)
		PrintBorderPoints(outb,F);
	FreeField(&F);	
	return 0;
}


