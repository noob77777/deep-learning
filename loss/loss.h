#ifndef LOSS
#define LOSS

#include "../matrix/matrix.h"
#include <math.h>

class LossFunction {
public:
	virtual float cost(Matrix A, Matrix Y) = 0;
	virtual Matrix derivative(Matrix A, Matrix Y) = 0;
};

class BinaryCrossEntropyLoss: public LossFunction {
	float cost(Matrix A, Matrix Y) {
		assert(A.is_shape_equal(Y));
		float res = 0;
		for(int i = 0; i < A.n; i++) {
			for(int j = 0; j < A.m; j++) {
				res += -(Y[i][j] * log(A[i][j]) + (1 - Y[i][j]) * log(1 - A[i][j]));
			}
		}
		return res / A.m;
	}

	Matrix derivative(Matrix A, Matrix Y) {
		Matrix res = ((Y/A) - ((Y-1)/(A-1))) * -1.0 / A.m;
		return res;
	}
};

#endif
