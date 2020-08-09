#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#include "./common.h"

void multiply(float **a, float **b, float **c, int n) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			c[i][j] = 0;
			for(int k = 0; k < n; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}


int main() {
	int fd = open("./testdata", O_RDONLY);

	int n;
	read(fd, (void *)&n, 4);
	
	printf("%d\n", n);

	float **a = (float **)malloc(n*sizeof(float *));
	float **b = (float **)malloc(n*sizeof(float *));
	float **c = (float **)malloc(n*sizeof(float *));
	for(int i = 0; i < n; i++) {
		a[i] = (float *)malloc(n*4);
		b[i] = (float *)malloc(n*4);
		c[i] = (float *)malloc(n*4); 
	}

	read_data(fd, a, n);
	read_data(fd, b, n);

	multiply(a, b, c, n);

	int fdw = open("./baseresult", O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);

	write_data(fdw, c, n);

	close(fdw);
	close(fd);

	return 0;
}
