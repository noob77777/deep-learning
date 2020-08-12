#include <bits/stdc++.h>
#include <fstream>

#include "../matrix/matrix.h"
#include "../layer/layer.h"
#include "../loss/loss.h"
#include "../neuralnetwork/neuralnetwork.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace std;

int main() {
    //
    //Matrix

    Matrix matrix = Matrix(5, 5, 'n');
    {
        ofstream ofs("matrix");
        boost::archive::text_oarchive oa(ofs);
        oa << matrix;
    }

    Matrix::print(matrix);

    Matrix newmatrix;
    {
        ifstream ifs("matrix");
        boost::archive::text_iarchive ia(ifs);
        ia >> newmatrix;
    }

    Matrix::print(newmatrix);
    cout << endl;

    //
    //Layer

    SigmoidLayer L(5, 2, 0.01, 1.0);
    L.forward_propagation(matrix);
    {
        ofstream ofs("layer");
        boost::archive::text_oarchive oa(ofs);
        oa << L;
    }

    cout << L.get_regularization_cost() << ' ';

    SigmoidLayer newL;
    {
        ifstream ifs("layer");
        boost::archive::text_iarchive ia(ifs);
        ia >> newL;
    }

    cout << newL.get_regularization_cost() << endl;

    //
    //Layer *

    Layer * Lptr = new ReluLayer(5, 2, 0.01, 1.0);
    Lptr->forward_propagation(matrix);
    {
        ofstream ofs("layer_pointer");
        boost::archive::text_oarchive oa(ofs);
        oa << Lptr;
    }

    cout << Lptr->get_regularization_cost() << ' ';

    Layer * newLptr;
    {
        ifstream ifs("layer_pointer");
        boost::archive::text_iarchive ia(ifs);
        ia >> newLptr;
    }

    cout << newLptr->get_regularization_cost() << endl;

    //
    //NeuralNetwork -> /neuralnetwork/neuralnetwork_test.cpp

    return 0;
}
