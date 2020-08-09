#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#define nax 5050

float buffer[nax];

int main() {
	int fd = open("./testdata", O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);

	int n;
	scanf("%d", &n);
	write(fd, (void *)&n, 4);

	for(int count = 0; count < 2; count++) {
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < n; j++) {
				buffer[j] = (1.0 * rand()) / RAND_MAX;
			}
			write(fd, (void *)buffer, 4*n);
		}
	}

	close(fd);

	return 0;
}
