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
#include <time.h>
#include <getopt.h>
#include "sudoku.h"
#include "sudoku_parser.h"
#include "sudoku_generator.h"
#include "sudokusolver.h"

int gss_verbose=1;
void PrintHeader()
{
	printf("  ____ ____ ____\n");
	printf(" / ___/ ___/ ___|  Generic Sudoku Solver\n");
	printf("| |  _\\___ \\___ \\  (c) Bart Pieters 2020\n");
	printf("| |_| |___) |__) |\n");
	printf(" \\____|____/____/\n");
	printf("Compiled with %d bit int support\n(handles block sizes up to %d elements)\n\n", (int)(8*sizeof(MASKINT)), (int)(8*sizeof(MASKINT)));
	printf("*******************************************\n");
	printf("This software is provided as is, without \n");
	printf("any warrenty. Consider yourself lucky if it\nworks.\n");
	printf("*******************************************\n");

}

SSTATE ProcessSudoku(Sudoku S, int verbose, int maxsol, int limitlevel, int STRAT, int *maxlevel)
{
	int v, ns, j;
	MASKINT **sol;
	clock_t t1, t2;
	SSTATE state;
	v=S_UK(S);
	if (verbose>1)
	{
		S_Print(S, stdout);
		printf("Sudoku with %d unknowns\n", v);
	}
	sol=malloc(sizeof(MASKINT *));
	t1=clock();
	state=Solve(&S, maxlevel, &sol, &ns, maxsol, limitlevel, STRAT);
	t2=clock();
	switch(state)
	{
		case UNSOLVED:
			if (verbose>0)
				fprintf(stderr, "Could not solve the sudoku!\n");
			break;
		case SOLVED:
			if (verbose)
			{
				if (verbose>1)
				{
					printf("Solved in %e s at level %d\n", ((double)(t2-t1))/CLOCKS_PER_SEC, *maxlevel);
					S_Print(S, stdout);
				}
			}
			break;
		case MULTISOL:
			if ((ns>=maxsol)&&verbose)
				fprintf(stderr, "Warning, sudoku has at least %d solutions. Stopped searching for more\n", ns);
			else if (verbose)
				fprintf(stderr, "Warning, %d solutions found\n", ns);
			if (verbose|1)
			{
				MASKINT *dummy;
				printf("Found %d solutions in %e s\n", ns, ((double)(t2-t1))/CLOCKS_PER_SEC);

				dummy=S.M;
				if (1)
				{
					for (j=0;j<ns;j++)
					{
						fprintf(stdout,"Solution %d:\n", j);
						S.M=sol[j];
						S_Print(S, stdout);
						free(sol[j]);
					}
				}
				S.M=dummy;
			}
			else
				for (j=0;j<ns;j++)
					free(sol[j]);
			break;
		case INVALID:
		default:
			fprintf(stderr, "Sudoku is invalid\n");
	}
	free(sol);
	return state;
}

int Fill(Sudoku S, int optfill)
{
	if (optfill)
		return ResolveConflicts(&S);
	else
		return FillEmptySudoku(&S);
}

void Help()
{
	printf("Usage:\ngss <options> <sudoku-file>\n\n");
	printf("Options:\n");
	printf("--help           [-h]  Print this short help\n");
	printf("--quiet          [-q]  Minimize output\n");
	printf("--verbose        [-v]  Maximize output\n");
	printf("--compact-format [-c]  Specify the compact sudoku format\n");
	printf("--native-format  [-n]  Specify sudokus in the native format\n");
	printf("--puzzle-out     [-p]  output file for the unsolved puzzle\n");
	printf("--solution-out   [-s]  output file for the solution\n");
	printf("--gss-out        [-G]  output file in the native format\n");
	printf("--limit-level    [-l]  set the maximum level for the used solution strategies\n");
	printf("--generate       [-g]  generate a sudoku. You need to specify a template sudoku for this option\n");
	printf("--template       [-t]  Provide a sudoku template\n");
	printf("--analyze       [-a]  Analyze a sudoku structure\n");
	printf("--strategies     [-S]  Select solving strategies\n");
	printf("--fill           [-f]  Fill an empty sudoku using backtracking\n");
	printf("--optfill        [-F]  Experimental filler using optimization\n");
}



int main(int argc, char **argv)
{
	int limitlevel=0;
	int level, fill=0, optfill=0;
	static int compact=0;
	int STRAT=0;
	FILE *pf=stdout; /* puzzle out */
	FILE *sf=stdout; /* solution out */
	FILE *gssf=NULL; /* gss out */
	int gen=0, maxsol=2, analyze=0, n_puzzles=0;
	char *template_sudoku=NULL;
	Sudoku S;
	int c;
	while (1)
	{
		static struct option long_options[] =
		{
			{"help",                no_argument, 0, 'h'},	/* simple help */
			{"quiet",               no_argument, 0, 'q'},	/* regulate the amout of output */
			{"verbose",             no_argument, 0, 'v'},	/* regulate the amout of output */
			{"compact-format",      no_argument, 0, 'c'},	/* compact format for standard 9x9 sudokus only (one line per soduku, one character per cell) */
			{"native-format",       no_argument, 0, 'n'},	/* gss format, with this format you can specify the sudoku size and structure and output format */
			{"puzzle-out",    required_argument, 0, 'p'},	/* outfile for puzzles */
			{"solution-out",  required_argument, 0, 's'},	/* outfile for corresponding solutions */
			{"gss-out   ",    required_argument, 0, 'G'},	/* output puzzle in native gss format (if you want the solution just run gss on it) */
			{"limit-level",   required_argument, 0, 'l'},	/* limit the level fo the logic solver (1-block size) */
			{"generate",      required_argument, 0, 'g'},	/* generate a sudoku after a template */
			{"analyze",             no_argument, 0, 'a'},	/* analyze a sudoku specified as template */
			{"template",      required_argument, 0, 't'},	/* template soduku input */
			{"strategies",    required_argument, 0, 'S'},	/* specify strategies elim,hidden,inter,brute*/
			{"fill",       			no_argument, 0, 'f'},	/* try to fill the template sudoku (rather than solving it). Per default it uses a backtracking filler */
			{"optfill",       		no_argument, 0, 'F'},	/* try to fill the template sudoku (rather than solving it). Use the optimizer to fill the sudoku (i.e. start with a wrong solution and try to make it better till it is right) */
			// {"n_puzzles",     required_argument, 0, 'N'},	/* try to fill the template sudoku (rather than solving it). Use the optimizer to fill the sudoku (i.e. start with a wrong solution and try to make it better till it is right) */

			{0, 0, 0, 0}
		};
		/* getopt_long stores the option index here. */
		int option_index = 0;
		c = getopt_long (argc, argv, "hqvcnp:s:G:l:g:at:S:fF:N",long_options, &option_index);
		/* Detect the end of the options. */
		if (c == -1)
			break;
		switch (c)
		{
			case 'h':
				Help();
				exit(0);
				break;
			case 'p':
				if (!optarg)
				{
					fprintf(stderr, "Error: --puzzle-out requires an output file\n");
					return 1;
				}
				if ((pf=fopen(optarg, "w"))==NULL)
				{
					fprintf(stderr, "cannot open file %s\n", optarg);
					exit(1);
				}
				break;
			case 's':
				if (!optarg)
				{
					fprintf(stderr, "Error: --solution-out requires an output file\n");
					return 1;
				}
				if ((sf=fopen(optarg, "w"))==NULL)
				{
					fprintf(stderr, "cannot open file %s\n", optarg);
					exit(1);
				}
				break;
			case 'G':
				if (!optarg)
				{
					fprintf(stderr, "Error: --gss-out requires an output file\n");
					return 1;
				}
				if ((gssf=fopen(optarg, "w"))==NULL)
				{
					fprintf(stderr, "cannot open file %s\n", optarg);
					exit(1);
				}
				break;
			case 'l':
				if (!optarg)
				{
					fprintf(stderr, "Error: --limit-level requires a maximum level\n");
					return 1;
				}
				limitlevel=atoi(optarg);
				break;
			case 'g':
				if (!optarg)
				{
					fprintf(stderr, "Error: --generate requires a level for the sudoku\n");
					return 1;
				}
				gen=atoi(optarg);
				if (gen<=0)
				{
					gen=0;
					// fprintf(stderr, "Error level should be larger than 0\n");
				}
				break;
			case 't':
				if (!optarg)
				{
					fprintf(stderr, "Error: --template requires template sudoku input file\n");
					return 1;
				}
				template_sudoku=optarg;
				break;
			case 'S':
			{
				char *p;
				if (!optarg)
				{
					fprintf(stderr, "Error: --strategies requires comma-separated strategy specifiers (elim, hidden, inter, brute)\n");
					return 1;
				}
				/* missing code */
				p=optarg;
				while (*p)
				{
					if (strncmp(p, "elim", 4)==0)
						STRAT|=(1<<MASK);
					else if (strncmp(p, "hidden", 6)==0)
						STRAT|=(1<<MASKHIDDEN);
					else if (strncmp(p, "inter", 5)==0)
						STRAT|=(1<<MASKINTER);
					else if (strncmp(p, "brute", 5)==0)
						STRAT|=(1<<BRUTE);
					else
					{
						fprintf(stderr, "Unknown strategy %s\n", p);
						exit(1);
					}
					while ((*p)&&(*p!=','))
						p++;
					if (*p==',')
						p++;
				}
				break;
			}
			case 'F':
				optfill=1;
			case 'f':
				fill=1;
				break;
			case 'q':
				gss_verbose=0;
				break;
			case 'a':
				analyze=1;
				break;
			case 'v':
				gss_verbose=2;
				break;
			case 'c':
				compact=1;
				break;
			case 'n':
				compact=0;
				break;
			case '?':
				/* getopt_long already printed an error message. */
				break;
			default:
				abort ();
		}
	}

	if ((optind < argc)||(gen)||(analyze)||(fill))
    {
		clock_t t1, t0, ts, te;
		int files=0;
		int processed=0, solved=0, unsolved=0, invalid=0, multisol=0;
		S_init();

		if (STRAT==0)
		{
			STRAT|=(1<<MASK);
			STRAT|=(1<<MASKHIDDEN);
			STRAT|=(1<<MASKINTER);
			STRAT|=(1<<BRUTE);
		}

		ts=clock();

		if (gen)
		{
			int j=0, g, k=0;
			int Gps;
			double dGps;

			if (!template_sudoku)
			{
				fprintf(stderr, "Error: please sepcify a sudoku template using the --template option\n");
				return 1;
			}
			if (gss_verbose)
				printf("* Generate sudoku according to template: --\n");
			S=S_Read(template_sudoku);
			files++;
			if (fill)
			{
				if (gss_verbose)
				{
					printf("Filling an empty sudoku:");
					if (S.BS>16)
						printf("Filling of large sudokus may take a long time\n");
				}

				/* fill it */
				if (Fill(S, optfill)!=0)
				{
					fprintf(stderr, "Error: could not fill the sudoku\n");
					return 1;
				}
				else
					printf("\t Success\n");

			}
			else
			{
				if (gss_verbose)
					printf("Solving Template Sudoku\n");
				switch(ProcessSudoku(S, gss_verbose, maxsol, limitlevel, STRAT, &level))
				{
					case UNSOLVED:
						unsolved++;
						break;
					case SOLVED:
						solved++;
						break;
					case MULTISOL:
						multisol++;
						break;
					case INVALID:
					default:
						invalid++;
				}
				processed++;
			}

			// if (gen>S.BS)
			// {
			// 	gen=S.BS;
			// 	printf("The maximum level for this sudoku is %d\n", S.BS);
			// }
			// if ((gen==S.BS)&&((SDO(BRUTE))==0))
			// {
			// 	gen=S.BS-1;
			// 	printf("The maximum level for this sudoku is %d (with current selected strategies)\n", S.BS-1);
			// }
			for(int i=0;i<gen;i++)
			{

				Randomize(&S);
				S_Print(S, sf);
				printf("\n");
			}

			// t0=clock();
			// g=GenerateSudoku(&S, gen, STRAT);
			// if (g<0)
			// {
			// 	S_free(&S);
			// 	exit(0);
			// }
			// t1=clock();
			// dGps=CLOCKS_PER_SEC/((double)(t1-t0));
			// Gps=CLOCKS_PER_SEC/(t1-t0);
			// printf("Generating a level %d sudoku at %.3f [Sudokus/s]", gen, dGps);
			// if (Gps<1)
			// 	Gps=1;
			// fflush(stdout);
			// if (g!=gen)
			// {
			// 	ProcessSudoku(S, 0, 1, limitlevel, STRAT, &level);
			// 	g=0;
			// 	while ((gen!=GenerateSudoku(&S, gen, STRAT)))
			// 	{
			// 		ProcessSudoku(S, 0, 1, limitlevel, STRAT, &level);
			// 		if (j%Gps==0)
			// 		{
			// 			if (k%4==0)
			// 			{
			// 				t1=clock();
			// 				dGps=(double)(j+2)*CLOCKS_PER_SEC/((double)(t1-t0));
			// 				Gps=(j+2)*CLOCKS_PER_SEC/(t1-t0);
			// 				printf("\rGenerating a level %d sudoku at %.3f [Sudokus/s]   \b\b\b",gen, dGps);
			// 				if (Gps<1)
			// 					Gps=1;
			// 				fflush(stdout);
			// 			}
			// 			else
			// 				putchar('.');
			// 			fflush(stdout);
			// 			k++;
			// 		}
			// 		j++;
			// 	}
			// }
			// printf("\n");
			//
			// printf("Generated %d sudokus to find one of level %d\n", j+1, gen);
			// S_Print(S, pf);


			if (gssf)
				GssOut(S, gssf, "|.|");
			S_free(&S);
			if (gss_verbose)
				printf("-------------------------------------------\n\n");
		}
		else if (fill)
		{
			if (!template_sudoku)
			{
				fprintf(stderr, "Error: please specify a sudoku template using the --template option\n");
				return 1;
			}
			S=S_Read(template_sudoku);
			files++;
			if (gss_verbose)
			{
				printf("* Filling an empty sudoku: ----------------\n");
				if (S.BS>16)
					printf("This may take a long time\n");
			}

			/* fill it */
			if (Fill(S,optfill)==0)
			{
				S_Print(S, sf);
				if (gss_verbose)
					printf("Success\n-------------------------------------------\n\n");
			}
			else
			{
				S_Print(S, sf);
				if (gss_verbose)
					printf("Failed\n-------------------------------------------\n\n");
			}
			if (gssf)
				GssOut(S, gssf, "|.|");
			S_free(&S);
		}
		while (optind < argc)
		{
			if (compact)
			{

				int nprocessed=0, nsolved=0, nunsolved=0, ninvalid=0, nmultisol=0, nguess=0, no_guess=0;
				FILE *fin;
				S=S_InitStdSudoku(compact);
				fin=OpenCompactMultiSudokuFile(argv[optind++]);
				files++;
				t0=clock();
				while (feof(fin)==0)
				{
					if (!S_ReadNextCompactStandard(fin, &S))
						break;
					S_Print(S,pf);
					switch(ProcessSudoku(S, gss_verbose, maxsol, limitlevel, STRAT, &level))
					{
						case UNSOLVED:
							unsolved++;
							nunsolved++;
							break;
						case SOLVED:
							solved++;
							nsolved++;
							S_Print(S,sf);
							break;
						case MULTISOL:
							multisol++;
							nmultisol++;
							break;
						case INVALID:
						default:
							invalid++;
							ninvalid++;
					}
					nprocessed++;
					processed++;
					nguess+=GUESS;
					no_guess+=(GUESS==0);
				}
				fclose(fin);
				t1=clock();
				printf("Results for sudokus in %s:\n", argv[optind-1]);
				printf("%d sudokus in %e s (%e sudokus/s):\n", nprocessed, ((double)(t1-t0))/CLOCKS_PER_SEC, (double)(nprocessed)*CLOCKS_PER_SEC/((double)(t1-t0)));
				printf("\t%d sudokus solved\n", nsolved);
				printf("\t%d guesses (%f/sudoku)\n", nguess, (double)nguess/(double)nprocessed);
				printf("\t%d sudokus solved without guesses\n", no_guess);
				printf("\t%d sudokus failed to solve\n", nunsolved);
				printf("\t%d sudokus with multiple solutions\n", nmultisol);
				printf("\t%d sudokus invalid\n", ninvalid);
				S_free(&S);
			}
			else
			{
				S=S_Read(argv[optind++]);
				files++;
// 				S_Print(S,pf);
				switch(ProcessSudoku(S, gss_verbose, maxsol, limitlevel, STRAT, &level))
				{
					case UNSOLVED:
						unsolved++;
						break;
					case SOLVED:
						solved++;
						S_Print(S,sf);
						break;
					case MULTISOL:
						multisol++;
						break;
					case INVALID:
					default:
						invalid++;
				}
				processed++;
				S_free(&S);

			}
		}
		te=clock();
		if (files>1)
		{
			printf("Totals\n%d files analyzed:\n", files);
			printf("%d sudokus in %e s (%e sudokus/s):\n", processed, ((double)(te-ts))/CLOCKS_PER_SEC, (double)(processed)*CLOCKS_PER_SEC/((double)(te-ts)));
			printf("\t%d sudokus solved\n", solved);
			printf("\t%d sudokus failed to solve\n", unsolved);
			printf("\t%d sudokus with multiple solutions\n", multisol);
			printf("\t%d sudokus invalid\n", invalid);
		}
	}
    else
	{
		fprintf(stderr, "Error: no input files found\n");
		exit(1);
	}

	if (pf!=stdout)
		fclose(pf);
	if (sf!=stdout)
		fclose(sf);

	if (gssf)
		fclose(gssf);
	return 0;

}
