#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <pthread.h>

#define THREAD_COUNT 5

struct thread_args
{
	int id;
};

void *test_function(void *args)
{
	printf("Return from %d\n", ((struct thread_args *)args)->id);
	return NULL;
}

int main(int argc, char *argv[])
{
	pthread_t t[THREAD_COUNT];

	for (int i = 0; i < THREAD_COUNT; i++)
	{
		struct thread_args *ta = (struct thread_args *)malloc(4);
		ta->id = i;
		int s = pthread_create(&t[i], NULL, test_function, ta);
	}

	for (int i = 0; i < THREAD_COUNT; i++)
	{
		void *res;
		int s = pthread_join(t[i], &res);
	}

	return 0;
}
