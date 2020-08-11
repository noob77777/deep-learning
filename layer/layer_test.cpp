#include <bits/stdc++.h>
#include "layer.h"

using namespace std;

int main() {

	int m = 1000;
	float learning_rate = 0.05;
	float lambda = 0.1;

	Matrix X = Matrix(32, m, 'u');
	Matrix Y = Matrix(1, m);

	Layer *L1 = new SigmoidLayer(32, 16, learning_rate, lambda);
	Layer *L2 = new SigmoidLayer(16, 1, learning_rate, lambda);

	for(int i = 0; i < 100; i++) {
		Matrix A1 = L1->forward_propagation(X);
		Matrix A2 = L2->forward_propagation(A1);

		Matrix dA1 = L2->backward_propagation(A2);
		L1->backward_propagation(dA1);

		Matrix::print(Matrix::sum(A2, 1));
	}

	return 0;
}
