#include <bits/stdc++.h>
#include "model.h"

using namespace std;

#define LEARNING_RATE 0.005
#define M 1000
#define T 100

int main(int argc, char const *argv[]) {

	Matrix X_train = Matrix(2, M, true);
	Matrix Y_train = Matrix(1, M);

	Matrix X_test = Matrix(2, T, true);
	Matrix Y_test = Matrix(1, T);

	for(int i = 0; i < M; i++) {
		if(X_train[0][i] + X_train[1][i] < 1.0) Y_train[0][i] = 1.0;
	}

	for(int i = 0; i < T; i++) {
		if(X_test[0][i] + X_test[1][i] < 1.0) Y_test[0][i] = 1.0;
	}

	NeuralNetwork nn = NeuralNetwork();
	nn.add_layer(new ReluLayer(2, 64, LEARNING_RATE));
	nn.add_layer(new ReluLayer(64, 8, LEARNING_RATE));
	nn.add_layer(new ReluLayer(8, 4, LEARNING_RATE));
	nn.add_layer(new ReluLayer(4, 4, LEARNING_RATE));
	nn.add_layer(new SigmoidLayer(4, 1, LEARNING_RATE));
	nn.add_loss_function(new BinaryCrossEntropyLoss());

	for(int i = 0; i < 10; i++) {
		float cost = nn.train_batch(X_train, Y_train);
		cout << "Cost after iteration " << ((i+1)*100) << ": " << cost << endl;
	}

	Matrix Y_pred = nn.predict(X_test);
	Matrix Y_pred_train = nn.predict(X_train);

	float test_accuracy = 0;
	for(int i = 0; i < T; i++) {
		float y_pred = Y_pred[0][i] > 0.5 ? 1 : 0;
		test_accuracy += (y_pred == Y_test[0][i]) ? 1 : 0;
	}

	test_accuracy /= T;
	cout << "Test Accuracy: " << test_accuracy << endl;

	float train_accuracy = 0;
	for(int i = 0; i < M; i++) {
		float y_pred_train = Y_pred_train[0][i] > 0.5 ? 1 : 0;
		train_accuracy += (y_pred_train == Y_train[0][i]) ? 1 : 0;
	}

	train_accuracy /= M;
	cout << "Train Accuracy: " << train_accuracy << endl;
	
	return 0;
}
