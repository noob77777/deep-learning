#include <fcntl.h>
#include <unistd.h>

#define nax 5050

float buffer[nax];

void read_data(int fd, float **a, int n)
{
	for (int i = 0; i < n; i++)
	{
		read(fd, (void *)buffer, 4 * n);
		for (int j = 0; j < n; j++)
		{
			a[i][j] = buffer[j];
		}
	}
}

void write_data(int fd, float **a, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			buffer[j] = a[i][j];
		}
		write(fd, (void *)buffer, 4 * n);
	}
}

void print_matrix(float **a, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%f ", a[i][j]);
		}
		printf("\n");
	}
}

float get_error(float **a, float **b, int n)
{
	float result = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			result += (a[i][j] - b[i][j]) * (a[i][j] - b[i][j]);
		}
	}
	return result;
}
