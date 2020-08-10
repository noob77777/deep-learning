#ifndef LAYER
#define LAYER

#include "../matrix/matrix.h"
#include <math.h>

using namespace std;

class Layer {
protected:
	Matrix W, b, A_, Z, D;
	float lambda, keep_prob, learning_rate;

	void update_parameters(Matrix dW, Matrix db);

	Matrix get_expanded_b(int m) {
		Matrix b_ = Matrix(b.n, m);
		for(int i = 0; i < b.n; i++) {
			for(int j = 0; j < m; j++) {
				b_[i][j] = b[i][0];
			}
		}
		return b_;
	}

	virtual Matrix activation(Matrix Z) = 0;
	virtual Matrix backward_activation(Matrix A) = 0;

public:
	Layer(int l_, int l, float learning_rate, float lambda = 0, float keep_prob = 1.0) {
		W = Matrix(l, l_, 'n') * ((float)sqrt(2.0/l_));
		b = Matrix(l, 1);
		this->learning_rate = learning_rate;
		this->lambda = lambda;
		this->keep_prob = keep_prob;
	}
	
	Matrix forward_propagation(Matrix A_);

	Matrix backward_propagation(Matrix dA);

};

void Layer::update_parameters(Matrix dW, Matrix db) {
	W = W - (dW * learning_rate);
	b = b - (db * learning_rate);
}

Matrix Layer::forward_propagation(Matrix A_) {
	this->A_ = A_;
	int m = A_.m;

	Z = Matrix::dot(W, A_) + get_expanded_b(m);
	Matrix A = activation(Z);

	return A;
}

Matrix Layer::backward_propagation(Matrix dA) {
	Matrix dZ = dA * backward_activation(Z);
	Matrix dW = Matrix::dot(dZ, A_.transpose());
	Matrix db = Matrix::sum(dZ, 1);
	Matrix dA_ = Matrix::dot(W.transpose(), dZ);

	update_parameters(dW, db);

	return dA_;
}

class SigmoidLayer: public Layer {
	float sigmoid(float x) {
		return (1 / (1 + exp((double) -x)));
	}

	float sigmoid_derivative(float x) {
		return sigmoid(x) * (1 - sigmoid(x));
	}

protected:
	Matrix activation(Matrix Z) {
		Matrix res = Matrix(Z.n, Z.m);
		for(int i = 0; i < Z.n; i++) {
			for(int j = 0; j < Z.m; j++) {
				res[i][j] = sigmoid(Z[i][j]);
			}
		}
		return res;
	}
	Matrix backward_activation(Matrix A) {
		Matrix res = Matrix(A.n, A.m);
		for(int i = 0; i < A.n; i++) {
			for(int j = 0; j < A.m; j++) {
				res[i][j] = sigmoid_derivative(A[i][j]);
			}
		}
		return res;
	}
public:
	SigmoidLayer(int l_, int l, float learning_rate, float lambda = 0, float keep_prob = 1.0): 
		Layer(l_, l, learning_rate, lambda = 0, keep_prob) {
			;
	}
};

class ReluLayer: public Layer {
	float relu(float x) {
		return x > 0 ? x : 0;
	}

	float relu_derivative(float x) {
		if(x > 0) return 1;
		return 0;
	}

protected:
	Matrix activation(Matrix Z) {
		Matrix res = Matrix(Z.n, Z.m);
		for(int i = 0; i < Z.n; i++) {
			for(int j = 0; j < Z.m; j++) {
				res[i][j] = relu(Z[i][j]);
			}
		}
		return res;
	}
	Matrix backward_activation(Matrix A) {
		Matrix res = Matrix(A.n, A.m);
		for(int i = 0; i < A.n; i++) {
			for(int j = 0; j < A.m; j++) {
				res[i][j] = relu_derivative(A[i][j]);
			}
		}
		return res;
	}
public:
	ReluLayer(int l_, int l, float learning_rate, float lambda = 0, float keep_prob = 1.0): 
		Layer(l_, l, learning_rate, lambda = 0, keep_prob) {
			;
	}
};

#endif
