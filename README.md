# deep-learning
Fast neural networks implementation in C++

### Compile with build.sh
Uncomment required files for the build
```
./build.sh
```

### Prerequisites
Boost C++
```
sudo apt-get install libboost-all-dev
```
g++
```
sudo apt-get install g++
```
<br />

## Sample: *Building a neural network in 5 lines of code.*
```
NeuralNetwork nn = NeuralNetwork();
nn.add_layer(new ReluLayer(28 * 28, 128, LEARNING_RATE, LAMBDA));
nn.add_layer(new ReluLayer(128, 32, LEARNING_RATE, LAMBDA));
nn.add_layer(new SoftmaxLayer(32, 10, LEARNING_RATE, LAMBDA));
nn.add_loss_function(new SoftmaxCrossEntropyLoss());
```

### Checkout notebook branch
Test the implementation with an interactive jupyter notebook

<br />

# Documentation
## Building blocks of the model
- ### `class Matrix`
    `file:  deep-learning/matrix/matrix.h` 
    
    Supports all basic matrix operations and the ability to do fast matrix dot products.<br />
    All Inputs must be fed into the network using this class.<br />
    All internal calculations work on top of these `Matrix` objects.<br />
    
    **Constructor:**
    `Matrix(int n, int m, char rand_init)`
    
    *Initialize a `n x m` matrix with suitable random values.*<br />
    *`rand_init = 'u'` for uniform distribution between 0 and 1*<br />
    *`rand_init = 'n'` for normal distribution with mean 0 and variance 1*<br />
	  *default `rand_init = 0` for zero initialization.*<br />
  
    
- ### `class Layer`
    `file:  deep-learning/layer/layer.h`

    The Layer class does what its name says. It represents the layers of the neural<br />
    network. This is an abstract class and implements the key forward propagation,<br />
    backward propagation and optimization steps. Other layers must **extend** this class<br />
    with its own activation function by overriding the *`Matrix activation(Matrix Z);`* and<br />
    *`Matrix backward_activation(Matrix Z);`* methods.<br />
    
    **Constructor:**
    `Layer(int l_, int l, float learning_rate, float lambda, float beta1, float beta2)`
    
    *Initialize the layer with the given parameters.*<br />
    *`l_`: number of units in previous layer*<br />
    *`l`: number of units for the layer*<br />
    *`learning_rate`: step size for optimization algorithm*<br />
    *`lambda`: regularization hyperparameter*<br />
    *`beta1`: first order term for 'adam' optimizer*<br />
    *`beta2`: second order term for 'adam' optimizer*<br />
    
    **The following activations are already implemented:**
    1. `class SigmoidLayer: public Layer`
    2. `class ReluLayer: public Layer`
    3. `class SoftmaxLayer: public Layer`
        
    *Mathematical details of the forward and backward propagation steps are described later.*
    
- ### `class LossFunction`
    `file:  deep-learning/loss/loss.h`

    This is an abstract base class for all loss functions that can be implemented. Other loss<br />
    functions must implement the *`float cost(Matrix A, Matrix Y);`* and<br />
    *`Matrix derivative(Matrix A, Matrix Y);`* methods.<br />
    
     **The following loss functions are already implemented:**
    1. `class BinaryCrossEntropyLoss: public LossFunction`
    2. `class SoftmaxCrossEntropyLoss: public LossFunction`
    
    **NOTE:** SoftmaxCrossEntropyLoss derivative calculates the derivative of cross entropy loss
    multiplied with softmax activation layer.
    
    *Mathematical details of the forward and backward propagation steps are described later.*


## Putting it all together
- ### `class NeuralNetwork`
    `file:  deep-learning/neuralnetwork/neuralnetwork.h`
    
    **Methods:**
    1. `void add_layer(Layer * L);`<br />
        *Adds a fully connected layer to the network.*

    2. `void add_loss_function(LossFunction * J);`<br />
        *Adds target loss function to the network.*
        
    3. `float train_batch(Matrix X, Matrix Y, int num_iterations);`<br />
        *Trains weights for num_interations for given input and labels.*<br />
        *returns cost*
        
    4. `Matrix predict(Matrix X);`<br />
    
    5. `void save(string filename);`<br />
        *Saves the network and its state to file.*
        
    6. `void load(string filename);`<br />
        *Loads a pre-trained network from file.*
        
## Sample: *Logistic Regression*
```
NeuralNetwork nn = NeuralNetwork();
nn.add_layer(new SigmoidLayer(NUM_INPUTS, 1, LEARNING_RATE, LAMBDA));
nn.add_loss_function(new BinaryCrossEntropyLoss());
nn.train_batch(X_train, Y_train, 1000);
Matrix Y_pred = nn.predict(X_test);
```
