#include <bits/stdc++.h>
#include <math.h>
#include "matrix.h"

using namespace std;

int main()
{
	Matrix a = Matrix(10, 50, 'u');
	Matrix b = Matrix(50, 10, 'n');

	float error = 0;
	Matrix res, res2, res_test;

	//L2 norm, mean, square, sqroot
	float norm = 0, square = 0, sqroot = 0, mean1 = 0;
	res = a.square();
	res2 = a.sqroot();

	for (int i = 0; i < a.n; i++)
	{
		for (int j = 0; j < a.m; j++)
		{
			norm += pow(a[i][j], 2);
			square += pow(res[i][j] - pow(a[i][j], 2), 2);
			sqroot += pow(res2[i][j] - sqrt(a[i][j]), 2);
			mean1 += a[i][j];
		}
	}
	norm = sqrt(norm);
	mean1 /= (a.n * a.m);

	cout << "Matrix norm works correctly: " << (abs(norm - a.norm()) <= 1e-8 ? "Yes" : "No") << endl;
	cout << "Matrix square works correctly: " << (abs(square) <= 1e-8 ? "Yes" : "No") << endl;
	cout << "Matrix sqroot works correctly: " << (abs(sqroot) <= 1e-8 ? "Yes" : "No") << endl;
	cout << "Matrix uniform distribution works correctly: " << (abs(mean1 - 0.5) <= 1e-2 ? "Yes" : "No") << endl;

	float mean2 = 0;

	for (int i = 0; i < b.n; i++)
	{
		for (int j = 0; j < b.m; j++)
		{
			mean2 += b[i][j];
		}
	}
	mean2 /= (b.n * b.m);

	cout << "Matrix normal distribution works correctly: " << (abs(mean2) <= 1e-2 ? "Yes" : "No") << endl;

	//dot product
	error = 0;
	res = Matrix::dot(a, b);
	res_test = Matrix(a.n, b.m);

	for (int i = 0; i < a.n; i++)
	{
		for (int j = 0; j < b.m; j++)
		{
			for (int k = 0; k < a.m; k++)
			{
				res_test[i][j] += a[i][k] * b[k][j];
			}
			error += pow(res_test[i][j] - res[i][j], 2);
		}
	}

	cout << "Matrix dot works correctly: " << (abs(error) <= 1e-8 ? "Yes" : "No") << endl;

	//sum
	float sum1 = 0, sum2 = 0;
	res = Matrix::sum(a, 1);
	for (int i = 0; i < a.n; i++)
	{
		float s = 0;
		for (int j = 0; j < a.m; j++)
		{
			s += a[i][j];
		}
		sum1 += pow(res[i][0] - s, 2);
	}
	res = Matrix::sum(a, 0);
	for (int j = 0; j < a.m; j++)
	{
		float s = 0;
		for (int i = 0; i < a.n; i++)
		{
			s += a[i][j];
		}
		sum2 += pow(res[0][j] - s, 2);
	}

	cout << "Matrix sum works correctly: " << (abs(sum1) + abs(sum2) <= 1e-7 ? "Yes" : "No") << endl;

	return 0;
}
