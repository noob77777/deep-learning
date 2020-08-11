#ifndef MODEL
#define MODEL

#include<fstream>

#include "../matrix/matrix.h"
#include "../layer/layer.h"
#include "../loss/loss.h"
#include <math.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace std;

class NeuralNetwork {
	vector<Layer *> layers;
	LossFunction * loss;

	friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & loss;
        ar & layers;
    }

public:
	void add_layer(Layer * L) {
		layers.push_back(L);
	}

	void add_loss_function(LossFunction * J) {
		loss = J;
	}

	float train_batch(Matrix X, Matrix Y, int num_iterations = 1000) {
		float cost;
		for(int _i = 0; _i < num_iterations; _i++) {
			float regularization_cost = 0;
			Matrix A = X;

			for(int i = 0; i < layers.size(); i++) {
				A = layers[i]->forward_propagation(A);
				regularization_cost += layers[i]->get_regularization_cost();
			}

			cost = loss->cost(A, Y) + regularization_cost;
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

	void save(string filename) {
		ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << (*this);
	}

	void load(string filename) {
		ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> (*this);
	}

};


#endif
