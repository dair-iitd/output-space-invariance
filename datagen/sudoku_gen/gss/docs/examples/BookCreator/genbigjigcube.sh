#!/bin/bash
NoCol=0
outf=bigjigcube.gss
pastell=4
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -h|--help)
	echo "Usage genjigsaw.sh [options]"
	echo "Options:"
	echo "    -h|--help:         Print this help"
	echo "    -p|--paleness:     Set a number for the paleness of the background color"
	echo "    -n|--no-color:     Do not use a background color"
	echo "    -o|--out:          Output filename"
	shift
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

tmpblocks1=$(mktemp)
tmpborder1=$(mktemp)
tmpblocks2=$(mktemp)
tmpborder2=$(mktemp)
tmpblocks3=$(mktemp)
tmpborder3=$(mktemp)

NN=192
JigSawMRF --rect 8 8 -p 4 -e $tmpblocks1 --outb $tmpborder1
JigSawMRF --rect 8 8 -p 4 -o 64 -e $tmpblocks2 --outb $tmpborder2
JigSawMRF --rect 8 8 -p 4 -o 128 -e $tmpblocks3 --outb $tmpborder3

# header
echo "<size> 192" > $outf
echo "<blocksize> 16" >> $outf
echo "<pattern>|.|" >> $outf
echo "<emptychar>0" >> $outf

# custom blocks
awk '{print "<block>",$0}' $tmpblocks1 >> $outf
awk '{print "<block>",$0}' $tmpblocks2 >> $outf
awk '{print "<block>",$0}' $tmpblocks3 >> $outf
echo "<block> 65 66 67 68 69 70 71 72 129 137 145 153 161 169 177 185 ">>$outf
echo "<block> 73 74 75 76 77 78 79 80 130 138 146 154 162 170 178 186 ">>$outf
echo "<block> 81 82 83 84 85 86 87 88 131 139 147 155 163 171 179 187 ">>$outf
echo "<block> 89 90 91 92 93 94 95 96 132 140 148 156 164 172 180 188 ">>$outf
echo "<block> 97 98 99 100 101 102 103 104 133 141 149 157 165 173 181 189 ">>$outf
echo "<block> 105 106 107 108 109 110 111 112 134 142 150 158 166 174 182 190 ">>$outf
echo "<block> 113 114 115 116 117 118 119 120 135 143 151 159 167 175 183 191 ">>$outf
echo "<block> 121 122 123 124 125 126 127 128 136 144 152 160 168 176 184 192 ">>$outf
echo "<block> 1 2 3 4 5 6 7 8 129 130 131 132 133 134 135 136 ">>$outf
echo "<block> 9 10 11 12 13 14 15 16 137 138 139 140 141 142 143 144 ">>$outf
echo "<block> 17 18 19 20 21 22 23 24 145 146 147 148 149 150 151 152 ">>$outf
echo "<block> 25 26 27 28 29 30 31 32 153 154 155 156 157 158 159 160 ">>$outf
echo "<block> 33 34 35 36 37 38 39 40 161 162 163 164 165 166 167 168 ">>$outf
echo "<block> 41 42 43 44 45 46 47 48 169 170 171 172 173 174 175 176 ">>$outf
echo "<block> 49 50 51 52 53 54 55 56 177 178 179 180 181 182 183 184 ">>$outf
echo "<block> 57 58 59 60 61 62 63 64 185 186 187 188 189 190 191 192 ">>$outf
echo "<block> 1 9 17 25 33 41 49 57 65 73 81 89 97 105 113 121 ">>$outf
echo "<block> 2 10 18 26 34 42 50 58 66 74 82 90 98 106 114 122 ">>$outf
echo "<block> 3 11 19 27 35 43 51 59 67 75 83 91 99 107 115 123 ">>$outf
echo "<block> 4 12 20 28 36 44 52 60 68 76 84 92 100 108 116 124 ">>$outf
echo "<block> 5 13 21 29 37 45 53 61 69 77 85 93 101 109 117 125 ">>$outf
echo "<block> 6 14 22 30 38 46 54 62 70 78 86 94 102 110 118 126 ">>$outf
echo "<block> 7 15 23 31 39 47 55 63 71 79 87 95 103 111 119 127 ">>$outf
echo "<block> 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 ">>$outf
# start of sudoku
echo "<sudoku>" >> $outf	
# start of asy script
# sudoku array, initialize empty

echo "import three;" >> $outf
echo "currentprojection=perspective(20,20,20);" >> $outf
echo "size(16cm,16cm);" >> $outf

echo -n "int[] f={" >> $outf	
i=1
while [ "$i" -lt "192" ]
do
	echo -n "|.|," >> $outf
	let "i=i+1"
done
echo "|.|};" >> $outf
echo "path3 cell_vx = (0,0,0)--(1,0,0)--(1,0,1)--(0,0,1)--cycle;">>$outf
echo "path3 cell_vy = (0,0,0)--(0,1,0)--(0,1,1)--(0,0,1)--cycle;">>$outf
echo "path3 cell_h = (0,0,0)--(0,1,0)--(1,1,0)--(1,0,0)--cycle;">>$outf

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
	for el in $(cat $tmpblocks1)
	do

		if [ "$nbl" -eq 0 ]
		then
			echo "r=(1/pastell+1)*(pastell+sin(2*pi*$block/16));" >> $outf
			echo "g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*$block/16));" >> $outf
			echo "b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*$block/16));" >> $outf
		fi
		echo "draw(surface(shift(floor(($el-1)/8), 7.999, ($el-1)%8)*cell_vx), rgb(r,g,b),light=nolight);" >> $outf
		let "nbl=nbl+1"
		if [ "$nbl" -eq "16" ]
		then
			nbl=0;
			let "block=block+3"
		fi
	done
	block=1
	for el in $(cat $tmpblocks2)
	do

		if [ "$nbl" -eq 0 ]
		then
			echo "r=(1/pastell+1)*(pastell+sin(2*pi*$block/16));" >> $outf
			echo "g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*$block/16));" >> $outf
			echo "b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*$block/16));" >> $outf
		fi
		echo "draw(surface(shift(7.999, floor(($el-65)/8), ($el-65)%8)*cell_vy), rgb(r,g,b),light=nolight);" >> $outf
		let "nbl=nbl+1"
		if [ "$nbl" -eq "16" ]
		then
			nbl=0;
			let "block=block+3"
		fi
	done
	block=2
	for el in $(cat $tmpblocks3)
	do

		if [ "$nbl" -eq 0 ]
		then
			echo "r=(1/pastell+1)*(pastell+sin(2*pi*$block/16));" >> $outf
			echo "g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*$block/16));" >> $outf
			echo "b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*$block/16));" >> $outf
		fi
		echo "draw(surface(shift(floor(($el-129)/8), ($el-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);" >> $outf
		let "nbl=nbl+1"
		if [ "$nbl" -eq "16" ]
		then
			nbl=0;
			let "block=block+3"
		fi
	done
fi
# draw stuff
# boundary of the puzzle
echo "path3 g;" >> $outf
echo "g=(0, 8, 0) -- (0, 8, 8) -- (0, 0, 8) -- (8, 0, 8) -- (8, 0, 0) -- (8, 8, 0) -- cycle;">>$outf
echo "draw(g, black+linewidth(3pt));">>$outf
echo "g=(8, 8, 0) -- (8, 8, 8) -- (0, 8, 8);">>$outf
echo "draw(g, black+linewidth(3pt));">>$outf
echo "g=(8, 8, 8) -- (8, 0, 8);">>$outf
echo "draw(g, black+linewidth(3pt));">>$outf


# boundaries between the custom blocks
echo "// vertical plane at y=8" >> $outf
echo -n "real[] A={" >> $outf	
tr '\n' ' ' < $tmpborder1 |tr '\t' ',' >> $outf
echo "};" >> $outf
echo "real x;">>$outf
echo "real y;">>$outf
echo "real z;">>$outf
echo "path3 g;">>$outf
echo "for (int i = 0; i < A.length/2; ++i) {">>$outf
echo "	x=A[2*i];">>$outf
echo "	z=A[2*i+1];">>$outf
echo "	if (x-floor(x)>0.1) // horizontal line">>$outf
echo "		g=(floor(x),8,round(z)) -- (floor(x)+1,8,round(z));">>$outf
echo "	else">>$outf
echo "		g=(round(x), 8, floor(z)) -- (round(x), 8, floor(z)+1);">>$outf
echo "	draw(g, black+linewidth(3pt));">>$outf
echo "}">>$outf

echo "// vertical plane at x=8" >> $outf
echo -n "real[] B={" >> $outf	
tr '\n' ' ' < $tmpborder2 |tr '\t' ',' >> $outf
echo "};" >> $outf
echo "for (int i = 0; i < B.length/2; ++i) {">>$outf
echo "	y=B[2*i];">>$outf
echo "	z=B[2*i+1];">>$outf
echo "	if (y-floor(y)>0.1) // horizontal line">>$outf
echo "		g=(8, floor(y),round(z)) -- (8, floor(y)+1,round(z));">>$outf
echo "	else">>$outf
echo "		g=(8, round(y), floor(z)) -- (8, round(y), floor(z)+1);">>$outf
echo "	draw(g, black+linewidth(3pt));">>$outf
echo "}">>$outf
echo "// horizontal plane at z=8" >> $outf
echo -n "real[] C={" >> $outf	
tr '\n' ' ' < $tmpborder3 |tr '\t' ',' >> $outf
echo "};" >> $outf
echo "for (int i = 0; i < C.length/2; ++i) {">>$outf
echo "	x=C[2*i];">>$outf
echo "	y=C[2*i+1];">>$outf
echo "	if (x-floor(x)>0.1) // line parallel to y axis">>$outf
echo "		g=(floor(x),round(y),8) -- (floor(x)+1,round(y),8);">>$outf
echo "	else">>$outf
echo "		g=(round(x), floor(y), 8) -- (round(x), floor(y)+1, 8);">>$outf
echo "	draw(g, black+linewidth(3pt));">>$outf
echo "}">>$outf



echo "int k=0;">>$outf
echo "for (int s = 0;s<3;++s) {">>$outf
echo "	for (int i = 0; i < 8; ++i) {">>$outf
echo "		for (int j = 0; j < 8; ++j) {">>$outf
echo "			if (s==0) ">>$outf
echo "			{">>$outf
echo "				draw(shift(i, 8, j)*cell_vx, black+linewidth(1pt));">>$outf
echo "				if (f[k]>0)">>$outf
echo "					label(XZ()*string(f[k]), p = fontsize(20pt), (i+0.5,8,j+0.5), Embedded);">>$outf
echo "			}">>$outf
echo "			if (s==1) ">>$outf
echo "			{">>$outf
echo "				draw(shift(8, i, j)*cell_vy, black+linewidth(1pt));">>$outf
echo "				if (f[k]>0)">>$outf
echo "					label(YZ()*string(f[k]),p = fontsize(20pt), (8, i+0.5,j+0.5), Embedded);">>$outf
echo "			}">>$outf
echo "			if (s==2) ">>$outf
echo "			{">>$outf
echo "				draw(shift(i, j,8)*cell_h, black+linewidth(1pt));">>$outf
echo "				if (f[k]>0) // there must be an easier way to make the numbers in the top plane like this">>$outf
echo "					label(XY()*string(f[k]),p = fontsize(20pt), (i+0.5,j+0.5,8), Embedded);">>$outf
echo "			}">>$outf
echo "			k=k+1;">>$outf
echo "		}">>$outf
echo "	}">>$outf
echo "}">>$outf

rm $tmpblocks1
rm $tmpborder1
rm $tmpblocks2
rm $tmpborder2
rm $tmpblocks3
rm $tmpborder3

