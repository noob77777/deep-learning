#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#include "./common.h"

int main()
{
	int fd = open("./testdata", O_RDONLY);

	int n;
	read(fd, (void *)&n, 4);

	printf("%d\n", n);

	float **a = (float **)malloc(n * sizeof(float *));
	float **b = (float **)malloc(n * sizeof(float *));
	for (int i = 0; i < n; i++)
	{
		a[i] = (float *)malloc(n * 4);
		b[i] = (float *)malloc(n * 4);
	}

	int fd1 = open("./baseresult", O_RDONLY);
	int fd2 = open("./fastresult", O_RDONLY);

	read_data(fd1, a, n);
	read_data(fd2, b, n);

	printf("%.10f\n", get_error(a, b, n));

	close(fd1);
	close(fd2);

	return 0;
}
