#include <bits/stdc++.h>
#include <fstream>

#include "./neuralnetwork/neuralnetwork.h"

//hyperparameters
#define LEARNING_RATE 0.02
#define LAMBDA 0.1

#define BATCH_ITERATION 50

using namespace std;

int main() {
	ifstream ifs("data.ssv");
	int M; ifs >> M;

	Matrix X = Matrix(2, M);
	Matrix Y = Matrix(1, M);

	for(int i = 0; i < M; i++) {
		ifs >> X[0][i] >> X[1][i];
		ifs >> Y[0][i];
	}

	NeuralNetwork nn = NeuralNetwork();
	nn.add_layer(new ReluLayer(2, 32, LEARNING_RATE, LAMBDA));
	nn.add_layer(new ReluLayer(32, 16, LEARNING_RATE, LAMBDA));
	nn.add_layer(new ReluLayer(16, 16, LEARNING_RATE, LAMBDA));
	nn.add_layer(new ReluLayer(16, 8, LEARNING_RATE, LAMBDA));
	nn.add_layer(new SigmoidLayer(8, 1, LEARNING_RATE, LAMBDA));
	nn.add_loss_function(new BinaryCrossEntropyLoss());

	for(int i = 0; i < 5; i++) {
		float cost = nn.train_batch(X, Y, BATCH_ITERATION);
		cout << "Cost after " << (i+1)*BATCH_ITERATION << " iterations: " << cost << endl;
	}

	float accuracy = 0;
	Matrix Y_pred = nn.predict(X);

	for(int i = 0; i < M; i++) {
		float y_pred = Y_pred[0][i] > 0.5 ? 1 : 0;
		accuracy += (y_pred == Y[0][i]) ? 1 : 0;
	} 

	accuracy /= M;

	cout << "Accuracy: " << accuracy << endl;

	Matrix X_plot = Matrix(2, 40000);
	int cnt = 0;
	for(int i = -100; i < 100; i++) {
		for(int j = -100; j < 100; j++) {
			float x = i * 1.0 / 100;
			float y = j * 1.0 / 100;
			X_plot[0][cnt] = x;
			X_plot[1][cnt] = y;
			cnt++;
		}
	}

	Matrix Y_plot = nn.predict(X_plot);

	ofstream ofs("result.ssv");

	for(int i = 0; i < 40000; i++) {
		ofs << X_plot[0][i] << ' ' << X_plot[1][i] << ' ' << Y_plot[0][i] << '\n';
	}

	cout << "Results saved." << endl;

	return 0;
}