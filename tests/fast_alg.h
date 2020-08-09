#include <immintrin.h>
#include <pthread.h>

void multiply_transpose(float **a, float **b, float **c, int n) {
	float **bT = (float **)malloc(n*sizeof(float *));
	for(int i = 0; i < n; i++) {
		bT[i] = (float *)malloc(n*sizeof(float));
	}

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			bT[i][j] = b[j][i];
		}
	}

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			c[i][j] = 0;
			for(int k = 0; k < n; k++) {
				c[i][j] += a[i][k] * bT[j][k];
			}
			for(int k = 0; k < 5; k++) {
				printf("%f %f\n", a[i][k], bT[j][k]);
			}
		}
	}

	for(int i = 0; i < n; i++) {
		free(bT[i]);
	}

	free(bT);
}


float dot_product_intrin(float *a, float *b, int n) {

	float total;

	__m128 num1, num2, num3, num4;

	num4 = _mm_setzero_ps();

  	for(int i = 0; i < n; i += 4) {
    	num1 = _mm_loadu_ps(a+i);
    	num2 = _mm_loadu_ps(b+i);
    	num3 = _mm_mul_ps(num1, num2);
    	num4 = _mm_add_ps(num4, num3);
  	}

  	num4 = _mm_hadd_ps(num4, num4);
  	num4 = _mm_hadd_ps(num4, num4);

  	_mm_store_ps(&total,num4);

  	return total;
}

void multiply_simd(float **a, float **b, float **c, int n) {
	float **bT = (float **)malloc(n*sizeof(float *));
	float **A = (float **)malloc(n*sizeof(float *));
	int N = n + ((4 - n%4) % 4);

	for(int i = 0; i < n; i++) {
		bT[i] = (float *)malloc(4*N);
	}
	for(int i = 0; i < n; i++) {
		A[i] = (float *)malloc(4*N);
	}

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			bT[i][j] = b[j][i];
			A[i][j] = a[i][j];
		}
		for(int j = n; j < N; j++) {
			bT[i][j] = A[i][j] = 0;
		}
	}


	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			c[i][j] = dot_product_intrin(A[i], bT[j], N);
		}
	}

	for(int i = 0; i < n; i++) {
		free(bT[i]); free(A[i]);
	}

	free(bT); free(A);

}



#define THREAD_COUNT 8

struct thread_args {
	float *a;
	float *b;
	int n;
};

void * dot_product_intrin_thread(void * args) {
  	float * a = ((struct thread_args *)args)->a;
  	float * b = ((struct thread_args *)args)->b;
  	int n = ((struct thread_args *)args)->n;

  	float * total = (float *)malloc(sizeof(float));

  	__m128 num1, num2, num3, num4;

  	num4 = _mm_setzero_ps();

  	for(int i = 0; i < n; i += 4) {
    	num1 = _mm_loadu_ps(a+i);
    	num2 = _mm_loadu_ps(b+i);
    	num3 = _mm_mul_ps(num1, num2);
    	num3 = _mm_hadd_ps(num3, num3);
    	num4 = _mm_add_ps(num4, num3);
  	}

  	num4 = _mm_hadd_ps(num4, num4);

  	_mm_store_ss(total,num4);

  	free((struct thread_args *)args);

  	return (void *)total;
}

void multiply_simd_thread(float **a, float **b, float **c, int n) {
	float **bT = (float **)malloc(n*sizeof(float *));
	float **A = (float **)malloc(n*sizeof(float *));
	int N = n + ((4 - n%4) % 4);

	for(int i = 0; i < n; i++) {
		bT[i] = (float *)malloc(4*N);
		A[i] = (float *)malloc(4*N);
	}

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			bT[i][j] = b[j][i];
			A[i][j] = a[i][j];
		}
		for(int j = n; j < N; j++) {
			bT[i][j] = A[i][j] = 0;
		}
	}

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j += THREAD_COUNT) {
			pthread_t t[THREAD_COUNT];
			struct thread_args * args = (struct thread_args *)malloc(sizeof(struct thread_args));
			for(int k = j, thread_count = 0; k < n && thread_count < THREAD_COUNT; k++, thread_count++) {
				struct thread_args * args = (struct thread_args *)malloc(sizeof(struct thread_args));
				args->a = A[i]; args->b = bT[k]; args->n = N;
				int s = pthread_create(&t[thread_count], NULL, dot_product_intrin_thread, args);
			}
			for(int k = j, thread_count = 0; k < n && thread_count < THREAD_COUNT; k++, thread_count++) {
				void * temp;
				int s = pthread_join(t[thread_count], &temp);
				c[i][k] = (*(float *)temp);
			}
		}
	}

	for(int i = 0; i < n; i++) {
		free(bT[i]); free(A[i]);
	}

	free(bT); free(A);

}
