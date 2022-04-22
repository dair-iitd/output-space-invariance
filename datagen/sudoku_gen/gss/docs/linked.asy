size(10cm);
int n = 3;

int N = n*n;
int[] f={0,0,0,0,0,0,3,0,2,0,7,0,0,0,0,0,0,0,0,6,0,0,5,8,0,4,0,0,0,5,0,1,0,0,0,3,0,0,0,0,2,0,5,0,0,6,2,0,0,0,0,0,0,0,8,0,0,0,0,9,0,0,0,1,0,0,0,0,4,0,6,0,0,0,0,0,7,0,1,0,0};

path cell = box((0,0),(1,1));
path supercell = box((0,0),(n,n));
int k=0;
int p=3;
int c;
pen[] P={palered,palegreen,paleblue,palecyan,pink,paleyellow,lightgrey,white,orange};
for (int i = 0; i < N; ++i) {
	for (int j = 0; j < N; ++j) {
		c=3*(i%3)+j%3;
		
		fill(shift(i, j)*cell, P[c]);
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


