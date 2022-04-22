import three;
settings.prc = false;
settings.render = 0;
size(10cm);
int n = 4;
currentprojection=perspective(20,20,20);

int sides = 3;
int N = 2*n;
int[] f={09,00,00,00,07,15,00,10,00,00,10,00,00,00,09,06,00,00,00,16,00,00,11,00,00,00,01,00,04,16,00,00,00,00,02,11,03,00,00,00,00,06,12,00,00,13,15,00,00,00,00,13,00,09,05,00,00,03,00,00,00,00,14,02,00,13,15,00,00,00,04,00,12,00,05,06,11,00,00,14,04,00,00,03,13,00,00,00,00,11,00,00,00,00,00,00,10,00,00,07,16,00,00,08,00,00,08,00,00,00,10,00,01,00,00,02,05,12,00,00,00,00,09,12,00,02,00,00,14,00,00,08,01,00,00,00,00,03,02,00,00,00,08,16,06,10,00,07,05,00,13,00,00,00,00,13,00,12,00,00,00,00,05,00,12,00,00,07,00,08,14,00,00,00,03,00,02,00,01,00,00,14,16,00,00,07,00,10,00,00,00,00};

// file in=input(out_n.dat).line();
// real[] f=in;

path3 cell_vx = (0,0,0)--(1,0,0)--(1,0,1)--(0,0,1)--cycle;
path3 supercell_vx = (0,0,0)--(n,0,0)--(n,0,n)--(0,0,n)--cycle;
path3 cell_vy = (0,0,0)--(0,1,0)--(0,1,1)--(0,0,1)--cycle;
path3 supercell_vy = (0,0,0)--(0,n,0)--(0,n,n)--(0,0,n)--cycle;
path3 cell_h = (0,0,0)--(0,1,0)--(1,1,0)--(1,0,0)--cycle;
path3 supercell_h = (0,0,0)--(0,n,0)--(n,n,0)--(n,0,0)--cycle;

int k=0;
for (int s = 0;s<sides;++s) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			if (s==0) 
			{
				draw(shift(i, N, j)*cell_vx, black+linewidth(1pt));
				if (f[k]>0)
					label(XZ()*string(f[k]), p = fontsize(13pt), (i+0.5,N,j+0.5), Embedded);
			}
			if (s==1) 
			{
				draw(shift(N, i, j)*cell_vy, black+linewidth(1pt));
				if (f[k]>0)
					label(YZ()*string(f[k]),p = fontsize(13pt), (N, i+0.5,j+0.5), Embedded);
			}
			if (s==2) 
			{
				draw(shift(i, j,N)*cell_h, black+linewidth(1pt));
				if (f[k]>0) // there must be an easier way to make the numbers in the top plane like this
					label(planeproject((1,1,1), (1,1,1))*rotate(135,(0,0,1))*string(f[k]),p = fontsize(13pt), (i+0.5,j+0.5,N), Embedded);
			}
			k=k+1;
		}
	}
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			if (s==0) 
				draw(shift(i*n, N, j*n)*supercell_vx, black+linewidth(2pt));
			if (s==1) 
				draw(shift(N,i*n, j*n)*supercell_vy, black+linewidth(2pt));
			if (s==2) 
				draw(shift(i*n, j*n,N)*supercell_h, black+linewidth(2pt));
		}
	}
			
	
}

