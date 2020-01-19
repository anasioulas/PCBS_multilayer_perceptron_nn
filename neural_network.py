import numpy as np

class NeuralNetwork(object):
    def __init__(self, inputs, outputs, layer_dimensions, learning_rate):
        """inputs: matrix of the inputs, each column contains the features of one input.
        We denote the number of the inputs by n_e, for the sake of presentation.

        outputs: matrix of the outputs, each column contains the corresponding output.

        layer_dimensions: (1xn) matrix of the dimensions of the layers, where n is the number of the layers and each
        entry gives the dimension of the corresponding layer.

        We randomly initialize the weights and the biases using normal distribution."""

        self.x = inputs
        self.y = outputs
        self.dim = layer_dimensions

        self.y_predicted = np.zeros(self.y.shape)

        #Assign random values from the standard normal distribution to weights and biases.
        self.weights = [np.random.randn(m,n) for n,m in zip(self.dim[:-1], self.dim[1:])]
        self.biases = [np.random.randn(n,1) for n in self.dim[1:]]

        #Dictionarys to save the activations and preactivations of each layer. For the input layer, this is a slight
        #abuse of notation, since there are not activations there. We do that to have uniform representation for all the
        #layers.
        #preactivations:input to the neuron, before we apply the activation function. Usually it is denoted by z.
        self.activations = {}
        self.preactivations = {}
        self.activations[0]=self.x

        self.lr = learning_rate


    def forward_propagation(self, x):
        """We do the forward pass for all the inputs (contained in x) together.

        Thus, z in each layer has as many as rows as the neurons of the layers and as many columns
        as the number of the examples in the inputs, ie n_e. The same holds for a as well.
        We use the ReLU activation function for all but the last layer.
        At the last layer we use the Sigmoid activation function.
        We save all the computed values from the forward pass.

        Returns the predicted output."""
        a = x
        index = len(self.dim) - 1 #index of output layer

        #The output layer is excluded from this for loop because it uses a different activation function.
        for i in range(1, index):
            z = np.dot(self.weights[i-1], a) + self.biases[i-1]
            self.preactivations[i] = z
            a = relu(z)
            self.activations[i] = a

        #Output layer's block
        z = np.dot(self.weights[index-1],a)+self.biases[index-1]
        self.preactivations[index]=z
        a = sigmoid(z)
        self.activations[index]=a
        self.y_predicted = a

        return self.y_predicted


    def backward_propagation(self):
        '''Executes the back propagation algorithm and updates the weights and the biases.'''

        cost_derivative_a ={}
        cost_derivative_z = {}
        cost_derivative_w = {}
        cost_derivative_b = {}

        #For the same reason as in forward propagation we have two blocks, one for the last layer (using sigmoid)
        #and one for the rest (using ReLU).

        #The index of the last layer.
        index = len(self.dim)-1

        #Block_1
        #1xn_e matrix, one for each example, where n_e is the number of examples.
        cost_derivative_a[index] = -np.divide(self.y, self.y_predicted)+np.divide(1-self.y, 1-self.y_predicted)

        #1xn_e matrix, one for each example
        cost_derivative_z[index] = cost_derivative_a[index] * sigmoid_derivative(self.preactivations[index])

        #n_sl -array, where n_sl is the number of the neurons in the second to last layer (last hidden layer)
        #Remember that activations for this layer is (n_sl x n_e) matrix.
        #Each element in the n_sl -array is the average over all n_e examples.
        cost_derivative_w[index] = np.dot(cost_derivative_z[index],self.activations[index-1].T )/self.y.shape[1]

        cost_derivative_b[index] = np.sum(cost_derivative_z[index])/self.y.shape[1]

        #Block_2
        for i in range(len(self.dim)-2,0,-1):
            cost_derivative_a[i] = np.dot(self.weights[i].T, cost_derivative_z[i+1])

            cost_derivative_z[i] = cost_derivative_a[i]*relu_derivative(self.preactivations[i])

            cost_derivative_w[i] = np.dot(cost_derivative_z[i],self.activations[i-1].T )/self.y.shape[1]

            cost_derivative_b[i] = np.sum(cost_derivative_z[i],axis=1)/self.y.shape[1]

        #update the weights and the biases
        for i in range(index):
            self.weights[i] -= self.lr * cost_derivative_w[i+1]

            #We transform cost_derivative_b[i] to a numpy matrix to match the self.biases[i] types/sizes
            cost_derivative_b[i+1] = np.mat(cost_derivative_b[i+1])
            cost_derivative_b[i+1] = cost_derivative_b[i+1].T

            self.biases[i] -= self.lr * cost_derivative_b[i+1]

        return


    def gradient_descent(self, iter):
        '''Executes the gradient descent algorithm for iter iterations.

        It prints the average loss (over all examples) every 1000 iterations.'''

        for i in range(iter):
            y_predicted = self.forward_propagation(self.x)
            loss = self.average_loss(y_predicted)
            self.backward_propagation()

            if i % 1000 == 0:
                print ("%i iterations completed. Average loss: %f" %(i, loss))
        return

    def average_loss(self,Y_predicted):
        """We use the Cross-Entropy loss function for binary classification.
        Returns the average loss over all the outputs."""
        loss = (1./self.y.shape[1])*(-np.dot(self.y, np.log(Y_predicted).T)-np.dot(1-self.y, np.log(1-Y_predicted).T))
        return loss

    def evaluate(self,x,y):
        '''Returns the accuracy of the network on inputs x and outputs y.'''

        y_predicted = self.forward_propagation(x)
        #Classify the predicted outputs as 1 if their value is greater than or equal to 0.5
        #and as 0 if it is less than 0.5.
        y_predicted[y_predicted >= 0.5] = 1
        y_predicted[y_predicted < 0.5] = 0

        #Find the indices of the correctly classified instances.
        common = np.where(y_predicted[0] == y[0])

        #Accuracy
        acc = len(common[0])/y.shape[1]

        return acc

#additional functions

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu_derivative(x):
    x[x>0] = 1
    x[x<=0] = 0
    return x

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))