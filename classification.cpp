#include <bits/stdc++.h>
#include <fstream>

#include "./neuralnetwork/neuralnetwork.h"

//hyperparameters
#define LEARNING_RATE 0.01
#define LAMBDA 0.1

#define BATCH_ITERATION 50

using namespace std;

int main() {
	int M = 1000;
	Matrix X = Matrix(2, M, 'u');
	Matrix Y = Matrix(4, M);

	for(int i = 0; i < X.m; i++) {
		if (X[0][i] < 0.5 && X[1][i] < 0.5) {
			Y[0][i] = 1;
		}
		else if (X[0][i] >= 0.5 && X[1][i] >= 0.5) {
			Y[1][i] = 1;
		}
		else if (X[0][i] < 0.5 && X[1][i] >= 0.5) {
			Y[2][i] = 1;
		}
		else {
			Y[3][i] = 1;
		}
	}
	

	NeuralNetwork nn = NeuralNetwork();
	nn.add_layer(new ReluLayer(2, 16, LEARNING_RATE, LAMBDA));
	nn.add_layer(new ReluLayer(16, 8, LEARNING_RATE, LAMBDA));
	nn.add_layer(new ReluLayer(8, 8, LEARNING_RATE, LAMBDA));
	nn.add_layer(new SoftmaxLayer(8, 4, LEARNING_RATE, LAMBDA));
	nn.add_loss_function(new SoftmaxCrossEntropyLoss());

	for(int i = 0; i < 5; i++) {
		float cost = nn.train_batch(X, Y, BATCH_ITERATION);
		cout << "Cost after " << (i+1)*BATCH_ITERATION << " iterations: " << cost << endl;
	}

	float accuracy = 0;
	Matrix Y_pred = nn.predict(X);

	for(int i = 0; i < M; i++) {
		int max_idx1 = -1;
		int max_idx2 = -1;
		float nax1 = -1;
		float nax2 = -1;
		for(int j = 0; j < 4; j++) {
			if (Y_pred[j][i] > nax1) {
				nax1 = Y_pred[j][i];
				max_idx1 = j;
			}
			if (Y[j][i] > nax2) {
				nax2 = Y_pred[j][i];
				max_idx2 = j;
			}
		}
		accuracy += (max_idx1 == max_idx2) ? 1 : 0;
	} 

	accuracy /= M;

	cout << "Accuracy: " << accuracy << endl;

	return 0;
}