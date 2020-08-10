#include <bits/stdc++.h>
#include <math.h>
#include "matrix.h"

using namespace std;

int main() {
	Matrix a = Matrix(100, 500, true);
	Matrix b = Matrix(500, 100, true);

	// Matrix::print(m);

	Matrix res = Matrix::dot(a, b);

	Matrix res_test = Matrix(a.n, b.m);

	float error = 0;
	for(int i = 0; i < a.n; i++) {
		for(int j = 0; j < b.m; j++) {
			for(int k = 0; k < a.m; k++) {
				res_test[i][j] += a[i][k] * b[k][j];
			}
			error += pow(res_test[i][j] - res[i][j], 2);
		}
	}

	cout << error;

	return 0;
}
