#!/bin/bash
# generate a template file for standard sudokus in gss in conjunction with asymptote of a given size
# the basse number defines the size and should be larger than 1 and smaller than 12 (gss does not support a block size of 144)
# the sudoku size is then base^2xbase^2, i.e. a standard 9x9 sudoku has base 3.
function usage
{
	echo Usage $0 \<base\>
	echo where base is an integer between 2 and 11
}

if [ ! -z "$1" ]
then
	if [ "$1" -eq "$1" ] 2>/dev/null
	then
		# $1 is a proper integer
		base=$1
	else
		usage
		exit 1
	fi
else
	base=3
fi

if [ "$base" -lt 2 ]
then
	usage
	exit 1
fi
if [ "$base" -ge 12 ]
then
	usage
	exit 1
fi

# block size bs
bs=$(($base*$base))
# sudoku size ss
ss=$(($bs*$bs))

# characters per element (used to scale fontsize a bit)
c=1
a=$bs
while [ $a -ge 10 ]
do
	a=$(($a/10))
	c=$(($c+1))
done
scale=1
fs=$((20 -  $scale*($c-1)))

printf "<size>%d\n" $ss
printf "<blocksize>%d\n" $bs
printf "<standardblocks>\n"
printf "<emptychar>0\n"
printf "<pattern>|.|\n"
printf "<sudoku>\n"

printf "int[] f={"
i=1
n=1;
while [ $i -le $base ]
do
	j=1
	while [ $j -le $base ]
	do
		k=1
		while [ $k -le $base ]
		do
			l=1
			while [ $l -le $base ]
			do
				num=$((($i-1) + ($k-1)*$base + ($l-1) + ($j-1)*$base));
				num=$(($num % $bs + 1))
				if [ $n == 1 ]
				then
					n=2
				else
					printf ","
				fi
				printf "|%0*d|" $c $num
				l=$(($l+1))
			done
			k=$(($k+1))
		done
		printf "\n\t"
		j=$(($j+1))
	done
	i=$(($i+1))
done

printf "};\n"
printf "size(%dcm);\n" $bs
printf "int n = %d;\n" $base
printf "int N = %d;\n" $bs
printf "path cell = box((0,0),(1,1));\n"
printf "path supercell = box((0,0),(n,n));\n"
printf "int k=0;\n"
printf "for (int i = 0; i < N; ++i) {\n"
printf "	for (int j = 0; j < N; ++j) {\n"
printf "		draw(shift(i, j)*cell, black+linewidth(0.5pt));\n"
printf "		if (f[k]>0)\n"
printf "			label(string(f[k]),p = fontsize(%dpt), (i+0.5,j+0.5));\n" $fs
printf "		k=k+1;\n"
printf "	}\n"
printf "}\n"
printf "for (int i = 0; i < n; ++i) {\n"
printf "	for (int j = 0; j < n; ++j) {\n"
printf "		draw(shift(n*i, n*j)*supercell, black+linewidth(2pt));\n"
printf "	}\n"
printf "}\n"
	
