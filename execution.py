from data_preparation import data_processing
from neural_network import NeuralNetwork
import numpy as np


np.random.seed(1)
#We call the data_processing() method of data_preparation.py
#and we get the training and validation sets (inputs and outputs).
x_train, y_train, x_val, y_val = data_processing()

#We set the dimension of the layers to be 9, 7, 7 and 1 for the
#input, first,second and third (output) layer respectively.
layer_dims  = [9, 7, 7, 1]

#We set the learning rate to be 0.05
learning_rate = 0.05


#We create an instance of our neural_network class.
nn = NeuralNetwork(x_train, y_train, layer_dims, learning_rate)

#We execute gradient descend algorith on this nn for 40000 iterations.
nn.gradient_descent(40000)

#Evaluate the performance of the neural network on the training and validation set.

#First, the training set.
acc_train = nn.evaluate(x_train, y_train)
print("Accuracy on the training set is: " + str(acc_train))

#Now, the validation set.
acc_val = nn.evaluate(x_val,y_val)
print("Accuracy on the validation set is: " + str(acc_val))