#!/bin/bash
NoCol=0
outf=jigsaw.gss
N=9
pastell=4
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -h|--help)
	echo "Usage genjigsaw.sh [options]"
	echo "Options:"
	echo "    -h|--help:         Print this help"
	echo "    -s|--size:         Set the edge length of the square field"
	echo "    -p|--paleness:     Set a number for the paleness of the background color"
	echo "    -n|--no-color:     Do not use a background color"
	echo "    -o|--out:          Output filename"
	shift
	;;	
    -s|--size)	
	N=$2;
	shift # past argument
	shift # past value
	;;
    -p|--paleness)
    pastell="$2"
    shift # past argument
    shift # past value
    ;;
    -n|--no-color)
    NoCol=1
    shift # past argument
    ;;
    -o|--out)
    outf="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    echo "Unknown option \"$1\""
    exit
    ;;
esac
done

tmpblocks=$(mktemp)
tmpborder=$(mktemp)


let "NN=N*N"
JigSawMRF --rect $N $N -p $N -e $tmpblocks --outb $tmpborder

# header
echo "<size> $NN" > $outf
echo "<blocksize> $N" >> $outf
echo "<rowblocks>" >> $outf
echo "<colblocks>" >> $outf
echo "<pattern>|.|" >> $outf
echo "<emptychar>0" >> $outf

# custom blocks
awk '{print "<block>",$0}' $tmpblocks >> $outf


# start of sudoku
echo "<sudoku>" >> $outf	

# start of asy script
# sudoku array, initialize empty
echo -n "int[] f={" >> $outf	
i=1
while [ "$i" -lt "$NN" ]
do
	echo -n "|.|," >> $outf
	let "i=i+1"
done
echo "|.|};" >> $outf

echo "size("$N"cm,"$N"cm);" >> $outf
echo "int N = $N;" >> $outf
echo "path cell = box((0,0),(1,1));" >> $outf
if [ "$NoCol" -eq "0" ]
then
	# I want light colored backgrounds
	block=0
	nbl=0;
	echo "int block=0;" >> $outf
	echo "real pastell=$pastell;"  >> $outf
	echo "real r;" >> $outf
	echo "real g;" >> $outf
	echo "real b;" >> $outf
	# use the same block data as before
	for el in $(cat $tmpblocks)
	do

		if [ "$nbl" -eq 0 ]
		then
			echo "r=(1/pastell+1)*(pastell+sin(2*pi*$block/$N));" >> $outf
			echo "g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*$block/$N));" >> $outf
			echo "b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*$block/$N));" >> $outf
		fi
		echo "fill(shift(floor(($el-1)/$N), ($el-1)%$N)*cell, rgb(r,g,b));" >> $outf
		let "nbl=nbl+1"
		if [ "$nbl" -eq "$N" ]
		then
			nbl=0;
			let "block=block+1"
		fi
	done
fi
# draw stuff
# boundary of the puzzle
echo "draw((0,0) -- (N,0), black+linewidth(3pt));" >> $outf
echo "draw((N,0) -- (N,N), black+linewidth(3pt));" >> $outf
echo "draw((N,N) -- (0,N), black+linewidth(3pt));" >> $outf
echo "draw((0,N) -- (0,0), black+linewidth(3pt));" >> $outf

# boundaries between the custom blocks
echo -n "real[] B={" >> $outf	
tr '\n' ' ' < $tmpborder |tr '\t' ',' >> $outf
echo "};" >> $outf
echo "int k=0;" >> $outf
echo "for (int i = 0; i < N; ++i) {" >> $outf
echo "	for (int j = 0; j < N; ++j) {" >> $outf
echo "		draw(shift(i, j)*cell, black+linewidth(0.5pt));" >> $outf
echo "		if (f[k]>0)" >> $outf
echo "			label(string(f[k]),p = fontsize(20pt), (i+0.5,j+0.5));" >> $outf
echo "		k=k+1;" >> $outf
echo "	}" >> $outf
echo "}" >> $outf
echo "real x;" >> $outf
echo "real y;" >> $outf
echo "for (int i = 0; i < B.length/2; ++i) {" >> $outf
echo "	x=B[2*i];" >> $outf
echo "	y=B[2*i+1];" >> $outf
echo "	if (x-floor(x)>0.1)" >> $outf
echo "		draw((floor(x),y) -- (floor(x)+1,y), black+linewidth(2pt));" >> $outf
echo "	else" >> $outf
echo "		draw((x, floor(y)) -- (x, floor(y)+1), black+linewidth(2pt));" >> $outf
echo "}" >> $outf

rm $tmpblocks
rm $tmpborder

