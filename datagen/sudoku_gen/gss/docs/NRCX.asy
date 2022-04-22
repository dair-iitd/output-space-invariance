size(10cm);
int n = 3;

int N = n*n;
int[] f={};
int[] f={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,8,0,0,0,0,0,0,5,7,0,0,0,2,0,0,0,0,0,0,9,0,0,4,0,0,0,0,0,0,0,0,0,3,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


path cell = box((0,0),(1,1));
path supercell = box((0,0),(n,n));
for (int i=0;i<2;++i) {
	for (int j = 0; j < 2; ++j) {
		fill(shift((n+1)*i+1, (n+1)*j+1)*supercell, mediumgrey);
		draw(shift((n+1)*i+1, (n+1)*j+1)*supercell, black+linewidth(1pt));
	}
}
for (int i=0;i<9;++i) {
	fill(shift(i, i)*cell, grey);
	fill(shift(i, N-i-1)*cell, grey);
}
int k=0;
for (int i = 0; i < N; ++i) {
	for (int j = 0; j < N; ++j) {
		draw(shift(i, j)*cell, black+linewidth(0.5pt));
		if (f[k]>0)
			label(string(f[k]),p = fontsize(20pt), (i+0.5,j+0.5));
		k=k+1;
	}
}
for (int i = 0; i < n; ++i) {
	for (int j = 0; j < n; ++j) {
		draw(shift(n*i, n*j)*supercell, black+linewidth(2pt));
	}
}
