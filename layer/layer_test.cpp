#include <bits/stdc++.h>
#include "layer.h"

using namespace std;

int main() {

	int m = 1000;
	float learning_rate = 0.05;
	float lambda = 0.1;

	Matrix X = Matrix(32, m, 'u');
	Matrix Y = Matrix(1, m);

	Layer *L1 = new ReluLayer(32, 16, learning_rate, lambda);
	Layer *L2 = new SoftmaxLayer(16, 8, learning_rate, lambda);
	Layer *L3 = new SigmoidLayer(8, 1, learning_rate, lambda);

	float cost;
	for(int i = 0; i < 1000; i++) {
		Matrix A1 = L1->forward_propagation(X);
		Matrix A2 = L2->forward_propagation(A1);
		Matrix A3 = L3->forward_propagation(A2);		

		cost = Matrix::sum(A3, 1)[0][0];

		Matrix dA2 = L3->backward_propagation(A3);
		Matrix dA1 = L2->backward_propagation(dA2);
		L1->backward_propagation(dA1);
	}

	cout << "Layers work correctly: " << (abs(cost) < 10 ? "Yes" : "No") << endl;

	return 0;
}
