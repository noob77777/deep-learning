#ifndef LAYER
#define LAYER

#include "../matrix/matrix.h"
#include <math.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>

using namespace std;

/**
 *	Abstract class Layer is the base class for all fully connected layers available.
 *	Extend this class to implement your own custom activation function.
 *	Overide the serialize, activation and backward_activation methods.
 */
class Layer
{
	friend class boost::serialization::access;

	// Internal method to serialize the class object.
	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &W;
		ar &b;
		ar &A_;
		ar &Z;
		ar &VW;
		ar &SW;
		ar &Vb;
		ar &Sb;
		ar &lambda;
		ar &learning_rate;
		ar &beta1;
		ar &beta2;
		ar &epsilon;
		ar &bias_counter;
	}

protected:
	/**
	 * 	W and b are the wieght and bias matrix for the layer.
	 * 	A_ is previous layer activation cached for back propagation.
	 * 	Z is given by WA_ + b.
	 * 	lambda is the regularization hyperparameter for L2 weight decay.
	 *
	 * 	Vw, Sw, Vb, Sb, beta1, beta2 and bias_counter are used to implement adam optimizer
	 * 	where variable names have there usual meaning.
	 *	See documentation for more details on the back propagation algorithm.
	 */

	Matrix W, b, A_, Z;
	Matrix VW, SW, Vb, Sb;
	float lambda, learning_rate;
	float beta1, beta2;
	float epsilon;
	int bias_counter;

	void update_parameters(Matrix dW, Matrix db);

	Matrix get_expanded_b(int m)
	{
		Matrix b_ = Matrix(b.n, m);
		for (int i = 0; i < b.n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				b_[i][j] = b[i][0];
			}
		}
		return b_;
	}

	virtual Matrix activation(Matrix Z) = 0;
	virtual Matrix backward_activation(Matrix A) = 0;

public:
	Layer()
	{
		;
	}

	Layer(int l_, int l, float learning_rate, float lambda = 0, float beta1 = 0.9, float beta2 = 0.999)
	{
		W = Matrix(l, l_, 'n') * ((float)sqrt(2.0 / l_));
		b = Matrix(l, 1);
		VW = Matrix(l, l_);
		SW = Matrix(l, l_);
		Vb = Matrix(l, 1);
		Sb = Matrix(l, 1);
		this->learning_rate = learning_rate;
		this->lambda = lambda;
		this->beta1 = beta1;
		this->beta2 = beta2;
		this->epsilon = 1e-8;
		this->bias_counter = 0;
	}

	Matrix forward_propagation(Matrix A_);

	Matrix backward_propagation(Matrix dA);

	float get_regularization_cost();
};

/**
 *	Implementation of adam optimization algorithm for back propagation.
 */
void Layer::update_parameters(Matrix dW, Matrix db)
{

	bias_counter++;

	VW = VW * beta1 + dW * (1.0 - beta1);
	Vb = Vb * beta1 + db * (1.0 - beta1);

	Matrix VW_corrected = VW / (1 - pow(beta1, bias_counter));
	Matrix Vb_corrected = Vb / (1 - pow(beta1, bias_counter));

	SW = SW * beta2 + dW.square() * (1.0 - beta2);
	Sb = Sb * beta2 + db.square() * (1.0 - beta2);

	Matrix SW_corrected = SW / (1 - pow(beta2, bias_counter));
	Matrix Sb_corrected = Sb / (1 - pow(beta2, bias_counter));

	W = W - ((VW_corrected / (SW_corrected.sqroot() + epsilon)) * learning_rate);
	b = b - ((Vb_corrected / (Sb_corrected.sqroot() + epsilon)) * learning_rate);
}

/**
 * 	Forward propagation step.
 *	A = g(WA_ + b), where g is activation function and A_ is previous layer activation.
 */
Matrix Layer::forward_propagation(Matrix A_)
{
	this->A_ = A_;
	int m = A_.m;

	Z = Matrix::dot(W, A_) + get_expanded_b(m);
	Matrix A = activation(Z);

	return A;
}

/**
 *	Backpropagation step.
 * 	Calculates gradients and updates the weights and biases with regularization cost.
 *	Returns gradient required for previous layer.
 */
Matrix Layer::backward_propagation(Matrix dA)
{
	int m = dA.m;

	Matrix dZ = dA * backward_activation(Z);
	Matrix dW = Matrix::dot(dZ, A_.transpose()) + (W * (lambda / m));
	Matrix db = Matrix::sum(dZ, 1);
	Matrix dA_ = Matrix::dot(W.transpose(), dZ);

	update_parameters(dW, db);

	return dA_;
}

float Layer::get_regularization_cost()
{
	int m = A_.m;
	float cost = (lambda / (2.0 * m)) * pow(W.norm(), 2);
	return cost;
}

/**
 * 	Implementation of sigmoid function.
 *	SigmoidLayer extends Layer.
 *	This format can be used to implement any activation function.
 */
class SigmoidLayer : public Layer
{
	float sigmoid(float x)
	{
		return (1 / (1 + exp((double)-x)));
	}

	float sigmoid_derivative(float x)
	{
		return sigmoid(x) * (1 - sigmoid(x));
	}

	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<Layer>(*this);
		;
	}

protected:
	Matrix activation(Matrix Z)
	{
		Matrix res = Matrix(Z.n, Z.m);
		for (int i = 0; i < Z.n; i++)
		{
			for (int j = 0; j < Z.m; j++)
			{
				res[i][j] = sigmoid(Z[i][j]);
			}
		}
		return res;
	}
	Matrix backward_activation(Matrix Z)
	{
		Matrix res = Matrix(Z.n, Z.m);
		for (int i = 0; i < Z.n; i++)
		{
			for (int j = 0; j < Z.m; j++)
			{
				res[i][j] = sigmoid_derivative(Z[i][j]);
			}
		}
		return res;
	}

public:
	SigmoidLayer()
	{
		;
	}

	SigmoidLayer(int l_, int l, float learning_rate, float lambda = 0, float beta1 = 0.9, float beta2 = 0.999) : Layer(l_, l, learning_rate, lambda, beta1, beta2)
	{
		;
	}
};

BOOST_CLASS_EXPORT_GUID(SigmoidLayer, "SigmoidLayer")

/**
 *	Implementation of 'relu' function.
 */
class ReluLayer : public Layer
{
	float relu(float x)
	{
		return x > 0 ? x : 0;
	}

	float relu_derivative(float x)
	{
		if (x > 0)
			return 1;
		return 0;
	}

	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<Layer>(*this);
	}

protected:
	Matrix activation(Matrix Z)
	{
		Matrix res = Matrix(Z.n, Z.m);
		for (int i = 0; i < Z.n; i++)
		{
			for (int j = 0; j < Z.m; j++)
			{
				res[i][j] = relu(Z[i][j]);
			}
		}
		return res;
	}
	Matrix backward_activation(Matrix Z)
	{
		Matrix res = Matrix(Z.n, Z.m);
		for (int i = 0; i < Z.n; i++)
		{
			for (int j = 0; j < Z.m; j++)
			{
				res[i][j] = relu_derivative(Z[i][j]);
			}
		}
		return res;
	}

public:
	ReluLayer()
	{
		;
	}

	ReluLayer(int l_, int l, float learning_rate, float lambda = 0, float beta1 = 0.9, float beta2 = 0.999) : Layer(l_, l, learning_rate, lambda, beta1, beta2)
	{
		;
	}
};

BOOST_CLASS_EXPORT_GUID(ReluLayer, "ReluLayer")

/**
 * 	Implementation of SoftmaxLayer.
 * 	Notice differences in implementation of the activation and the backward_activation
 * 	methods.
 *	This layer only works correctly with SoftmaxCrossEntropyLoss.
 */
class SoftmaxLayer : public Layer
{
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<Layer>(*this);
	}

protected:
	Matrix activation(Matrix Z)
	{
		Matrix res = Matrix(Z.n, Z.m);
		for (int j = 0; j < Z.m; j++)
		{
			float denominator = 0;
			for (int i = 0; i < Z.n; i++)
			{
				denominator += exp(Z[i][j]);
			}
			for (int i = 0; i < Z.n; i++)
			{
				res[i][j] = exp(Z[i][j]) / denominator;
			}
		}
		return res;
	}

	Matrix backward_activation(Matrix Z)
	{
		// works only with SoftmaxCrossEntropy
		return Matrix(Z.n, Z.m) + 1;
	}

public:
	SoftmaxLayer()
	{
		;
	}

	SoftmaxLayer(int l_, int l, float learning_rate, float lambda = 0, float beta1 = 0.9, float beta2 = 0.999) : Layer(l_, l, learning_rate, lambda, beta1, beta2)
	{
		;
	}
};

BOOST_CLASS_EXPORT_GUID(SoftmaxLayer, "SoftmaxLayer")

#endif
