#ifndef MATRIX
#define MATRIX

#include <immintrin.h>
#include <iostream>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;

static inline float _mm256_reduce_add_ps(__m256 x) {
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

static inline float dot_product_intrin(float *a, float *b, int n) {
	float total;
	__m256 num1, num2, num3, num4;

	num4 = _mm256_setzero_ps();

  	for(int i = 0; i < n; i += 8) {
    	num1 = _mm256_loadu_ps(a+i);
    	num2 = _mm256_loadu_ps(b+i);
    	num3 = _mm256_mul_ps(num1, num2);
    	num4 = _mm256_add_ps(num4, num3);
  	}

  	total = _mm256_reduce_add_ps(num4);

  	return total;
}



default_random_engine generator;

class Matrix {
	vector<vector<float>> a;

	friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & n;
        ar & m;
        ar & a;
    }
	
public:
	int n; int m;
	
	Matrix(int n = 0, int m = 0, char rand_init = 0) {
		a = vector<vector<float>>(n, vector<float>(m));
		this->n = n;
		this->m = m;

		if (rand_init == 'u') {
			uniform_real_distribution<double> distribution(0.0, 1.0);

			for(int i = 0; i < n; i++) {
				for(int j = 0; j < m; j++) {
					a[i][j] = static_cast <float> (distribution(generator));
				}
			}
		} else if (rand_init == 'n') {
			normal_distribution<double> distribution(0.0, 1.0);

			for(int i = 0; i < n; i++) {
				for(int j = 0; j < m; j++) {
					a[i][j] = static_cast <float> (distribution(generator));
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

	bool is_shape_equal(Matrix b) {
		return (n == b.n && m == b.m);
	}

	Matrix operator+(float x);
	Matrix operator-(float x);
	Matrix operator*(float x);
	Matrix operator/(float x);

	Matrix operator+(Matrix b);
	Matrix operator-(Matrix b);
	Matrix operator*(Matrix b);
	Matrix operator/(Matrix b);

	Matrix transpose();
	Matrix square();
	Matrix sqroot();

	float norm();
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

Matrix Matrix::square() {
	Matrix A = Matrix(n, m);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A[i][j] = pow(a[i][j], 2);
		}
	}
	return A;
}

Matrix Matrix::sqroot() {
	Matrix A = Matrix(n, m);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A[i][j] = sqrt(a[i][j]);
		}
	}
	return A;
}

float Matrix::norm() {
	float res = 0;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			res += pow(a[i][j], 2);
		}
	}
	return sqrt(res);
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

	int N = b.n + ((8 - b.n%8) % 8);

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

#endif
