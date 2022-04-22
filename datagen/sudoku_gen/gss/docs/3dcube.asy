size(10cm);

// sudoku array
int[] f={00,00,00,00,09,00,00,13,00,00,06,00,00,00,04,00,06,00,13,00,00,07,00,00,00,05,10,00,00,15,16,00,00,00,00,00,05,00,00,00,00,00,00,09,00,08,11,00,00,16,00,00,14,15,00,00,00,00,00,00,04,00,01,00,08,01,04,07,10,13,00,00,00,16,00,03,06,00,00,12,00,00,10,00,00,00,00,11,15,00,02,12,01,00,00,00};

// a cell is 1x1
path cell = box((0,0),(1,1));

// a supercell is 4x4 (one block)
path supercell = box((0,0),(4,4));

// draw supercells and cells
int xoff=0;
int yoff=4;
// supercell 1 (left most supercell)
draw(shift(xoff, yoff)*supercell, black+linewidth(2pt));
// fill it with cells
for (int i=xoff;i<xoff+4; ++i) {
	for (int j=yoff;j<yoff+4; ++j) {
		draw(shift(i, j)*cell, black+linewidth(0.5pt));
	}
}

// move one supercell to the right and do the next one
xoff=4;
yoff=4;
draw(shift(xoff, yoff)*supercell, black+linewidth(2pt));
for (int i=xoff;i<xoff+4; ++i) {
	for (int j=yoff;j<yoff+4; ++j) {
		draw(shift(i, j)*cell, black+linewidth(0.5pt));
	}
}

// move one supercell to the right and do the next one
xoff=8;
yoff=4;
draw(shift(xoff, yoff)*supercell, black+linewidth(2pt));
for (int i=xoff;i<xoff+4; ++i) {
	for (int j=yoff;j<yoff+4; ++j) {
		draw(shift(i, j)*cell, black+linewidth(0.5pt));
	}
}


// move one supercell to the right and do the next one
xoff=12;
yoff=4;
draw(shift(xoff, yoff)*supercell, black+linewidth(2pt));
for (int i=xoff;i<xoff+4; ++i) {
	for (int j=yoff;j<yoff+4; ++j) {
		draw(shift(i, j)*cell, black+linewidth(0.5pt));
	}
}	



// make a supercell above the second supercell
xoff=4;
yoff=8;
draw(shift(xoff, yoff)*supercell, black+linewidth(2pt));
for (int i=xoff;i<xoff+4; ++i) {
	for (int j=yoff;j<yoff+4; ++j) {
		draw(shift(i, j)*cell, black+linewidth(0.5pt));
	}
}

// make a supercell below the second supercell
xoff=4;
yoff=0;
draw(shift(xoff, yoff)*supercell, black+linewidth(2pt));
for (int i=xoff;i<xoff+4; ++i) {
	for (int j=yoff;j<yoff+4; ++j) {
		draw(shift(i, j)*cell, black+linewidth(0.5pt));
	}
}


// add the glue-strips
xoff=0;
yoff=4;
draw((xoff, yoff)--(xoff+0.5,yoff-1), black+linewidth(0.5pt));
draw((xoff+0.5, yoff-1)--(xoff+4-0.5,yoff-1), black+linewidth(0.5pt));
draw((xoff+4-0.5, yoff-1)--(xoff+4,yoff), black+linewidth(0.5pt));

draw((xoff, yoff+4)--(xoff+0.5,yoff+4+1), black+linewidth(0.5pt));
draw((xoff+0.5, yoff+4+1)--(xoff+4-0.5,yoff+4+1), black+linewidth(0.5pt));
draw((xoff+4-0.5, yoff+4+1)--(xoff+4,yoff+4), black+linewidth(0.5pt));

xoff=8;
yoff=4;

draw((xoff, yoff)--(xoff+0.5,yoff-1), black+linewidth(0.5pt));
draw((xoff+0.5, yoff-1)--(xoff+4-0.5,yoff-1), black+linewidth(0.5pt));
draw((xoff+4-0.5, yoff-1)--(xoff+4,yoff), black+linewidth(0.5pt));

draw((xoff, yoff+4)--(xoff+0.5,yoff+4+1), black+linewidth(0.5pt));
draw((xoff+0.5, yoff+4+1)--(xoff+4-0.5,yoff+4+1), black+linewidth(0.5pt));
draw((xoff+4-0.5, yoff+4+1)--(xoff+4,yoff+4), black+linewidth(0.5pt));


xoff=12;
yoff=4;
draw((xoff, yoff)--(xoff+0.5,yoff-1), black+linewidth(0.5pt));
draw((xoff+0.5, yoff-1)--(xoff+4-0.5,yoff-1), black+linewidth(0.5pt));
draw((xoff+4-0.5, yoff-1)--(xoff+4,yoff), black+linewidth(0.5pt));

draw((xoff, yoff+4)--(xoff+0.5,yoff+4+1), black+linewidth(0.5pt));
draw((xoff+0.5, yoff+4+1)--(xoff+4-0.5,yoff+4+1), black+linewidth(0.5pt));
draw((xoff+4-0.5, yoff+4+1)--(xoff+4,yoff+4), black+linewidth(0.5pt));

draw((xoff+4, yoff+4)--(xoff+4+1,yoff+4-0.5), black+linewidth(0.5pt));
draw((xoff+4+1, yoff+4-0.5)--(xoff+4+1,yoff+0.5), black+linewidth(0.5pt));
draw((xoff+4+1, yoff+0.5)--(xoff+4,yoff), black+linewidth(0.5pt));


// Fill the cells in the correct order (i.e. in the order by which the elements appear in the block definition matrices
// i.e. from up-left to down-right
xoff=4;
yoff=8;
int k=0;
for (int j = 0; j < 4; ++j) {
	for (int i = 0; i < 4; ++i) {
		if (f[k]>0)
			label(string(f[k]),p = fontsize(13pt), (xoff+i+0.5,yoff+4-j-0.5));
		k=k+1;
	}
}

xoff=0;
yoff=4;
for (int j = 0; j < 4; ++j) {
	for (int i = 0; i < 16; ++i) {
		if (f[k]>0)
			label(string(f[k]),p = fontsize(13pt), (xoff+i+0.5,yoff+4-j-0.5));
		k=k+1;
	}
}

xoff=4;
yoff=0;
for (int j = 0; j < 4; ++j) {
	for (int i = 0; i < 4; ++i) {
		if (f[k]>0)
			label(string(f[k]),p = fontsize(13pt), (xoff+i+0.5,yoff+4-j-0.5));
		k=k+1;
	}
}


