#include <bits/stdc++.h>
#include <fstream>

#include "./mnist/read_mnist.hpp"
#include "./neuralnetwork/neuralnetwork.h"

//files
#define X_TRAIN "./mnist/train-images-idx3-ubyte"
#define Y_TRAIN "./mnist/train-labels-idx1-ubyte"
#define X_TEST "./mnist/t10k-images-idx3-ubyte"
#define Y_TEST "./mnist/t10k-labels-idx1-ubyte"

#define READ 60000
#define READ_TEST 10000

//hyperparameters
#define EPOCHS 1
#define BATCH_SIZE 1000
#define LEARNING_RATE 0.01
#define LAMBDA 0.1

#define BATCH_ITERATION 1

using namespace std;

int main() {
	//load
	vector<vector<float>> X_train = get_images(X_TRAIN, READ);
	vector<vector<float>> Y_train = get_labels(Y_TRAIN, READ);

	show_image(X_train[0]);
	cout << "Label: " << get_max_idx(Y_train[0]) << endl << endl;

	vector<vector<float>> X_test = get_images(X_TEST, READ_TEST);
	vector<vector<float>> Y_test = get_labels(Y_TEST, READ_TEST);


	//build 
	NeuralNetwork nn = NeuralNetwork();
	nn.add_layer(new ReluLayer(28 * 28, 128, LEARNING_RATE, LAMBDA));
	nn.add_layer(new ReluLayer(128, 64, LEARNING_RATE, LAMBDA));
	nn.add_layer(new ReluLayer(64, 32, LEARNING_RATE, LAMBDA));
	nn.add_layer(new SoftmaxLayer(32, 10, LEARNING_RATE, LAMBDA));
	nn.add_loss_function(new SoftmaxCrossEntropyLoss());


	//train
	int count = 0;
	for(int epoch = 0; epoch < EPOCHS; epoch++) {
		for(int batch = 0; batch < READ; batch += BATCH_SIZE) {
			Matrix X_batch = Matrix(28 * 28, BATCH_SIZE);
			Matrix Y_batch = Matrix(10, BATCH_SIZE);

			for(int i = 0; i < BATCH_SIZE; i++) {
				for(int j = 0; j < 28 * 28; j++) {
					X_batch[j][i] = X_train[i + batch][j];
				}
				for(int j = 0; j < 10; j++) {
					Y_batch[j][i] = Y_train[i + batch][j];
				}
			}

			float cost = nn.train_batch(X_batch, Y_batch, BATCH_ITERATION);
			
			if(count % 10 == 0) {
				cout << "Cost after " << count << " iterations: " << cost << endl;
			}

			count++;
		}
	}

	cout << endl;


	//train-accuracy
	float train_accuracy = 0;
	Matrix train_set = Matrix(READ, 28 * 28);
	for(int i = 0; i < READ; i++) {
		for(int j = 0; j < 28 * 28; j++) {
			train_set[i][j] = X_train[i][j];
		}
	}
	
	train_set = train_set.transpose();
	Matrix train_pred = nn.predict(train_set).transpose();

	for(int i = 0; i < READ; i++) {
		int a = get_max_idx(train_pred[i]);
		int y = get_max_idx(Y_train[i]);
		if (a == y) train_accuracy += 1.0;
	}

	train_accuracy /= READ;
	cout << "Train set accuracy: " << train_accuracy << endl;


	//test-accuracy
	float test_accuracy = 0;
	Matrix test_set = Matrix(READ_TEST, 28 * 28);
	for(int i = 0; i < READ_TEST; i++) {
		for(int j = 0; j < 28 * 28; j++) {
			test_set[i][j] = X_test[i][j];
		}
	}
	
	test_set = test_set.transpose();
	Matrix test_pred = nn.predict(test_set).transpose();

	for(int i = 0; i < READ_TEST; i++) {
		int a = get_max_idx(test_pred[i]);
		int y = get_max_idx(Y_test[i]);
		if (a == y) test_accuracy += 1.0;
	}

	test_accuracy /= READ_TEST;
	cout << "Test set accuracy: " << test_accuracy << endl << endl;


	//show stuff
	for(int sample = 0; sample < READ_TEST; sample += 2000) {
		show_image(X_test[sample]);
		cout << "Correct label: " << get_max_idx(Y_test[sample]) << endl;
		cout << "Predicted label: " << get_max_idx(test_pred[sample]) << endl << endl;
	}


	//save
	nn.save("mnist.neuralnetwork");	

	return 0;
}
