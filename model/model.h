#ifndef MODEL
#define MODEL

#include "../matrix/matrix.h"
#include "../layer/layer.h"
#include "../loss/loss.h"
#include <math.h>
#include <vector>

using namespace std;

class NeuralNetwork {
	vector<Layer *> layers;
	LossFunction * loss;

public:
	void add_layer(Layer * L) {
		layers.push_back(L);
	}

	void add_loss_function(LossFunction * J) {
		loss = J;
	}

	float train_batch(Matrix X, Matrix Y, int num_iterations = 1000) {
		float cost;
		for(int _i = 0; _i < num_iterations; _i++){
			Matrix A = X;
			for(int i = 0; i < layers.size(); i++) {
				A = layers[i]->forward_propagation(A);
			}

			cost = loss->cost(A, Y);
			Matrix dA = loss->derivative(A, Y);

			for(int i = layers.size() - 1; i >= 0; --i) {
				dA = layers[i]->backward_propagation(dA);
			}
		}

		return cost;
	}

	Matrix predict(Matrix X) {
		Matrix A = X;
		for(int i = 0; i < layers.size(); i++) {
			A = layers[i]->forward_propagation(A);
		}
		return A;
	}

};


#endif
