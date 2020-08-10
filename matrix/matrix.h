#include <immintrin.h>
#include <vector>
#include <iostream>

using namespace std;

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

class Matrix {
	vector<vector<float>> a;
	
public:
	int n; int m;
	
	Matrix(int n = 0, int m = 0, bool rand_init = false) {
		a = vector<vector<float>>(n, vector<float>(m));
		this->n = n;
		this->m = m;
		if(rand_init) {
			for(int i = 0; i < n; i++) {
				for(int j = 0; j < m; j++) {
					a[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				}
			}
		}
	}

	vector<float>& operator[](int i) {
		return a[i];
	}

	static void print(Matrix a) {
		for(int i = 0; i < a.n; i++) {
			for(int j = 0; j < a.m; j++) {
				cout << a[i][j] << ' ';
			}
			cout << '\n';
		}
	}

	static Matrix dot(Matrix a, Matrix b);

	static Matrix sum(Matrix a, int axis);

	Matrix operator+(float x);
	Matrix operator-(float x);
	Matrix operator*(float x);
	Matrix operator/(float x);

	Matrix operator+(Matrix b);
	Matrix operator-(Matrix b);
	Matrix operator*(Matrix b);
	Matrix operator/(Matrix b);

	Matrix transpose();
};

Matrix Matrix::operator+(float x) {
	Matrix A = Matrix(n, m);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A[i][j] = a[i][j] + x;
		}
	}
	return A;
}

Matrix Matrix::operator-(float x) {
	Matrix A = Matrix(n, m);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A[i][j] = a[i][j] - x;
		}
	}
	return A;
}

Matrix Matrix::operator*(float x) {
	Matrix A = Matrix(n, m);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A[i][j] = a[i][j] * x;
		}
	}
	return A;
}

Matrix Matrix::operator/(float x) {
	Matrix A = Matrix(n, m);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A[i][j] = a[i][j] / x;
		}
	}
	return A;
}

Matrix Matrix::operator+(Matrix b) {
	assert(n == b.n && m == b.m);
	Matrix A = Matrix(n, m);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A[i][j] = a[i][j] + b[i][j];
		}
	}
	return A;
}

Matrix Matrix::operator-(Matrix b) {
	assert(n == b.n && m == b.m);
	Matrix A = Matrix(n, m);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A[i][j] = a[i][j] - b[i][j];
		}
	}
	return A;
}

Matrix Matrix::operator*(Matrix b) {
	assert(n == b.n && m == b.m);
	Matrix A = Matrix(n, m);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A[i][j] = a[i][j] * b[i][j];
		}
	}
	return A;
}

Matrix Matrix::operator/(Matrix b) {
	assert(n == b.n && m == b.m);
	Matrix A = Matrix(n, m);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A[i][j] = a[i][j] / b[i][j];
		}
	}
	return A;
}

Matrix Matrix::transpose() {
	Matrix A = Matrix(m, n);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A[j][i] = a[i][j];
		}
	}
	return A;
}

Matrix Matrix::sum(Matrix a, int axis) {
	Matrix res;
	if(axis == 0) {
		res = Matrix(1, a.m);
		for(int i = 0; i < a.n; i++) {
			for(int j = 0; j < a.m; j++) {
				res[0][j] += a[i][j];
			}
		}
		return res;
	} else if(axis == 1) {
		res = Matrix(a.n, 1);
		for(int i = 0; i < a.n; i++) {
			for(int j = 0; j < a.m; j++) {
				res[i][0] += a[i][j];
			}
		}
	}
	return res;
}

Matrix Matrix::dot(Matrix a, Matrix b) {
	assert(a.m == b.n);
	
	float **Bt = new float*[b.m];
	float **A = new float*[a.n];

	int N = b.n + ((4 - b.n%4) % 4);

	for(int i = 0; i < b.m; i++) {
		Bt[i] = new float[N];
	}

	for(int i = 0; i < b.m; i++) {
		for(int j = 0; j < b.n; j++) {
			Bt[i][j] = b[j][i];
		}
		for(int j = b.n; j < N; j++) {
			Bt[i][j] = 0;
		}
	}

	for(int i = 0; i < a.n; i++) {
		A[i] = new float[N];
	}

	for(int i = 0; i < a.n; i++) {
		for(int j = 0; j < a.m; j++) {
			A[i][j] = a[i][j];
		}
		for(int j = a.m; j < N; j++) {
			A[i][j] = 0;
		}
	}

	Matrix c = Matrix(a.n, b.m);

	for(int i = 0; i < c.n; i++) {
		for(int j = 0; j < c.m; j++) {
			c[i][j] = dot_product_intrin(A[i], Bt[j], N);
		}
	}

	for(int i = 0; i < b.m; i++) {
		free(Bt[i]);
	}

	for(int i = 0; i < a.n; i++) {
		free(A[i]);
	}

	free(A); free(Bt);

	return c;
}
