#!/bin/bash
# generate a basic (ASCII formatting) template file for standard sudokus in gss of a given size
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

# characters per element
c=1
a=$bs
while [ $a -ge 10 ]
do
	a=$(($a/10))
	c=$(($c+1))
done

# build a vertical separation string for the sudoku table
hline="+"
# compute block width in characters
LW=$(($base*$c+$base+1))
i=1
while [ $i -le $base ]
do
	j=1
	while [ $j -le $LW ]
	do
		hline=$hline"-"
		j=$(($j+1))
	done
	hline=$hline"+"
	i=$(($i+1))
done
	

printf "<size>%d\n" $ss
printf "<blocksize>%d\n" $bs
printf "<standardblocks>\n"
printf "<sudoku>\n"
echo $hline
i=1
while [ $i -le $base ]
do
	j=1
	while [ $j -le $base ]
	do
		printf "| "
		k=1
		while [ $k -le $base ]
		do
			l=1
			while [ $l -le $base ]
			do
				num=$((($i-1) + ($k-1)*$base + ($l-1) + ($j-1)*$base));
				num=$(($num % $bs + 1))
				printf "%0*d " $c $num
				l=$(($l+1))
			done
			printf "| "
			k=$(($k+1))
		done
		printf "\n"
		j=$(($j+1))
	done
	echo $hline
	i=$(($i+1))
done
