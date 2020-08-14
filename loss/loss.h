#ifndef LOSS
#define LOSS

#include "../matrix/matrix.h"
#include <math.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>

const float ZERO = 1e-7;

class LossFunction {
	friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
    	;
    }

public:
	virtual float cost(Matrix A, Matrix Y) = 0;
	virtual Matrix derivative(Matrix A, Matrix Y) = 0;
};



class BinaryCrossEntropyLoss: public LossFunction {
	friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
    	ar & boost::serialization::base_object<LossFunction>(*this);
    }

public:
	float cost(Matrix A, Matrix Y) {
		assert(A.is_shape_equal(Y));
		float res = 0;
		for(int i = 0; i < A.n; i++) {
			for(int j = 0; j < A.m; j++) {
				res += -(Y[i][j] * log(max(A[i][j], ZERO)) + (1 - Y[i][j]) * log(max(1 - A[i][j], ZERO)));
			}
		}
		return res / A.m;
	}

	Matrix derivative(Matrix A, Matrix Y) {
		Matrix res = ((Y/(A + ZERO)) - ((Y-1)/(A - 1 + ZERO))) * -1.0 / A.m;
		return res;
	}
};

BOOST_CLASS_EXPORT_GUID(BinaryCrossEntropyLoss, "BinaryCrossEntropyLoss")



class SoftmaxCrossEntropyLoss: public LossFunction {
	friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
    	ar & boost::serialization::base_object<LossFunction>(*this);
    }

public:
	float cost(Matrix A, Matrix Y) {
		assert(A.is_shape_equal(Y));
		float res = 0;
		for(int i = 0; i < A.n; i++) {
			for(int j = 0; j < A.m; j++) {
				res += -(Y[i][j] * log(max(A[i][j], ZERO)));
			}
		}
		return res / A.m;
	}

	Matrix derivative(Matrix A, Matrix Y) {
		Matrix res = (A - Y) / A.m;
		return res;
	}
};

BOOST_CLASS_EXPORT_GUID(SoftmaxCrossEntropyLoss, "SoftmaxCrossEntropyLoss")

#endif
