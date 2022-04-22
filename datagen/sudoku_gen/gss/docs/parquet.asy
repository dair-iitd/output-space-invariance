size(10cm);
int n = 3;
int N = n*n;
int[] f={0,0,0,0,1,0,0,0,0,
		 0,3,0,0,0,0,0,0,8,
		 6,0,0,3,0,0,0,0,0,
		 0,9,0,0,0,0,0,0,0,
		 0,0,0,0,0,0,8,0,2,
		 0,0,0,0,4,0,5,9,0,
		 0,0,0,0,8,0,0,0,0,
		 0,0,0,0,0,0,0,0,5,
		 1,0,0,0,0,0,0,2,0};
path cell = box((0,0),(1,1));
path supercell = box((0,0),(4,4));
path ccell = box((0,0),(2,2));
path ucell = (0,0) -- (1,0) -- (1,1) -- (2,1) -- (2,2) -- (0,2) -- cycle;
path dcell = (0,0) -- (2,0) -- (2,2) -- (1,2) -- (1,1) -- (0,1) -- cycle;

int k=0;
int x;
int y;
for (int j=0;j<3;++j) {
	for (int i=0;i<3;++i) {
		x=i*4;
		y=j*4;		
		draw(shift(x, y)*cell, black+linewidth(0.5pt));
		if (f[k]>0)
			label(string(f[k]),p = fontsize(20pt), (x+0.5,y+0.5));
		k=k+1;
		
		x=x+1;
		draw(shift(x, y)*cell, black+linewidth(0.5pt));
		if (f[k]>0)
			label(string(f[k]),p = fontsize(20pt), (x+0.5,y+0.5));
		k=k+1;
		
		x=x+1;
		fill(shift(x, y)*dcell, lightgrey);
		draw(shift(x, y)*dcell, black+linewidth(0.5pt));
		if (f[k]>0)
			label(string(f[k]),p = fontsize(20pt), (x+1.5,y+0.5));
		k=k+1;
		
		x=i*4;
		y=y+1;			
		draw(shift(x, y)*cell, black+linewidth(0.5pt));
		if (f[k]>0)
			label(string(f[k]),p = fontsize(20pt), (x+0.5,y+0.5));
		k=k+1;
		
		x=x+1;		
		fill(shift(x, y)*ccell, mediumgrey);
		draw(shift(x, y)*ccell, black+linewidth(0.5pt));
		if (f[k]>0)
			label(string(f[k]),p = fontsize(20pt), (x+1,y+1));
		k=k+1;
		
		x=x+2;	
		y=y+1;		
		draw(shift(x, y)*cell, black+linewidth(0.5pt));
		if (f[k]>0)
			label(string(f[k]),p = fontsize(20pt), (x+0.5,y+0.5));
		k=k+1;
		
		x=i*4;		
		fill(shift(x, y)*ucell, lightgrey);
		draw(shift(x, y)*ucell, black+linewidth(0.5pt));
		if (f[k]>0)
			label(string(f[k]),p = fontsize(20pt), (x+0.5,y+1.5));
		k=k+1;
		
		
		x=x+2;
		y=y+1;
		draw(shift(x, y)*cell, black+linewidth(0.5pt));
		if (f[k]>0)
			label(string(f[k]),p = fontsize(20pt), (x+0.5,y+0.5));
		k=k+1;
		
		x=x+1;
		draw(shift(x, y)*cell, black+linewidth(0.5pt));
		if (f[k]>0)
			label(string(f[k]),p = fontsize(20pt), (x+0.5,y+0.5));
		k=k+1;
	}
}

for (int i=0;i<3;++i) {
	for (int j=0;j<3;++j) {	
		draw(shift(4*i, 4*j)*supercell, black+linewidth(2pt));
	}
}

