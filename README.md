Multilayer perceptron neural network from scratch
===========================

The goal of my project was to create a multilayer perceptron neural network from scratch, i.e. without using the standard deep learning libraries of Python like keras or pytorch.
More specifically, I restricted myself to using only numpy for the construction of the neural network. 

For the sake of illustration, I trained and evaluated the performance of my network on a specific data set (more details about that below). I used pandas and sklearn to prepare my data -the "from sratch" condition applied only to the construction of then neural network.

I created three scripts, namely [data_preparation.py](data_preparation.py), [neural_network.py](neural_network.py), and [execution.py](execution.py). The main script is the `execution.py`, which calls the other two. 

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Multilayer perceptron neural network from scratch](#multilayer-perceptron-neural-network-from-scratch)
    - [Preparation of the data](#preparation-of-the-data)
    - [The neural network](#the-neural-network)
    - [Execution script](#execution-script)
    - [Performance Results](#performance-results)
    - [Technical note](#technical-note)

    
    - [Previous & Gained Experience](#previous-&-gained-experience)

<!-- markdown-toc end -->


## Preparation of the data

I downloaded the Breast Cancer Wisconsin (Diagnostic) Data Set from <https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)>. More specifically, from 
<https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/> I downloaded the second file named "breast-cancer-wisconsin.data". I then converted it to a csv file, by renaming it as "breast-cancer-wisconsin.csv", so that my code can read it as a csv file. Then I created a script, namely [data_preparation.py](data_preparation.py), used for the preparation of the data. 

In this script, first I do some technical processing (excluding some instances with missing features and changing the values of classification in the data -2 and 4- to match standard values -0 and 1- used in binary classification in machine learning). Then I normalize the data by employing the min-max normalization. Finally, I split the data set into a training set and a validation set. 

The script `data_preparation.py`:

    def data_processing():
        """Manipulation of Breast Cancer Wisconsin (Diagnostic) Data Set.

        Initially, 699 instances (patients) with 9 features each.
        In the initial data set, a tumor is classified as benign (corresponding to number 2 at the last column)
        or malignant (corresponding to number 4 at the last column).

        Returns normalized training and validation sets.
        """

        from sklearn import preprocessing
        import pandas as pd

        #Read the file containing the data.
        df = pd.read_csv('wisconsin-cancer-dataset.csv',header=None)

        #Process the data.

        #There are some missing data in column 6. We exclude these rows.
        #There are 683 remaining rows with all the features included.
        df = df[~df[6].isin(['?'])]

        #In the data, last column (i.e. 10) is 2 for benign and 4 for malignant.
        #Change it to be 0 for benign and 1 for malignant.
        df.iloc[:,10].replace(2, 0,inplace=True)
        df.iloc[:,10].replace(4, 1,inplace=True)

        #Data normalization.
        names = df.columns[1:10]
        scaler = preprocessing.MinMaxScaler()
        scaled_df = scaler.fit_transform(df.iloc[:, 1:10])
        scaled_df = pd.DataFrame(scaled_df, columns=names)

        #Split data set into training and validation sets.
        #Training set (first 500 instances).
        x_train=scaled_df.iloc[0:500,:].values.transpose()
        y_train=df.iloc[0:500,10:].values.transpose()
        #Validation set.
        x_val=scaled_df.iloc[501:683,:].values.transpose()
        y_val=df.iloc[501:683,10:].values.transpose()

        return x_train, y_train, x_val, y_val


## The neural network

The script that constructs, trains and evaluates the neural network, namely [neural_network.py](neural_network.py), uses only numpy, as mentioned above. 

I used an object-oriented programming approach. My script implements all the standard functionalities of such a neural network, like `gradient descent` (which incorporates forward and `backward propagation`). 

Some of the choices I made were based on the specific data set. For example, since the data set concerns a binary classification problem, I used the standard `cross-entropy loss` function during the training phase. Yet, most of the code can be used for other data sets with very few (if any at all) modifications. 

Also, I used the `Rectified Linear Unit (ReLU)` as the activation function for the hidden layers and the `sigmoid` function as the activation function for the output layer.

Since I used a small data set, I did not use mini-batches, but instead I run forward and backward propagation on the whole training set (`batch gradient descent`).

Finally, in order to classify the instances and eventually evaluate the performance of my neural network, I used a `threshold` of 0.5, which means that instances with output value greater than or equal to 0.5 are classified as 1 and instances with output value less than 0.5 are classified as 0 -I remind that the sigmoid function gives values in (0,1).  

## Execution script

Finally, I created a script, namely [execution.py](execution.py), that combines the two previous scripts. 

This script first calls the [data_preparation.py](data_preparation.py) and receives the training and the validation sets. Then it sets some of the hyperparameters of the network to be instantiated. In the version uploaded I specifically use: 
* a learning rate of 0.05
* a 3-layer network with layer dimensions (9,7,7,1) (remember that the input layer is excluded from the counting of the number of the layers)

It then creates an instance of the NeuralNetwork class of the [neural_network.py](neural_network.py), runs the gradient_descent method for a number of iterations (passed to the gradient_descent method) and evaluates the performance of the resulting network. The gradient_descent method prints the average loss (over all the examples) every 1000 iterations.

Here is the script `execution.py`:

    from data_preparation import data_processing
    from neural_network import NeuralNetwork
    import numpy as np


    np.random.seed(1)
    #We call the data_processing() method of data_preparation.py
    #and we get the training and validation sets (inputs and outputs).
    x_train, y_train, x_val, y_val = data_processing()

    #We set the dimension of the layers to be 9, 15 and 1 for the
    #input, first and second (output) layer respectively.
    layer_dims  = [9, 7, 7, 1]

    #We set the learning rate to be 0.01
    learning_rate = 0.05


    #We create an instance of our neural_network class.
    nn = NeuralNetwork(x_train, y_train, layer_dims, learning_rate)

    #We execute gradient descend algorith on this nn for 65000 iterations.
    nn.gradient_descent(40000)

    #Evaluate the performance of the neural network on the training and validation set.

    #First, the training set.
    acc_train = nn.evaluate(x_train, y_train)
    print("Accuracy on the training set is: " + str(acc_train))

    #Now, the validation set.
    acc_val = nn.evaluate(x_val,y_val)
    print("Accuracy on the validation set is: " + str(acc_val))

## Performance Results

We present a few results of my program. 
For the following we used the `np.random.seed(1)`, so that the results are reproducable. 

* learning_rate = 0.05, layer_dimensions = (9,7,7,1), iterations = 40000
    * Accuracy on the training set: 1 
    * Accuracy of on the validation set: 0.989
   
* learning_rate = 0.06, layer_dimensions = (9,14,1), iterations = 60000
    * Accuracy on the training set: 0.996
    * Accuracy of on the validation set: 0.989
    
* learning_rate = 0.06, layer_dimensions = (9,5,5,5,1), iterations = 15000
    * Accuracy on the training set: 0.978
    * Accuracy of on the validation set: 0.994
    

## Technical note

I should mention that the use of the sigmoid function combined with the cross-entropy loss function creates somes issues with specific set of hyperparameters and usually after a big number of iterations. The problem is created by the fact that sigmoid gives outputs close to 0 or 1 which in turn, for example, might lead to log(0) in the computation of the loss. There are similar issues with division by a number very close to zero. There are ways to handle such issues. For instance, we can check when this is the case and add a very small epsilon, so that the log function (or the division) does not explode to infinity. Still, the network works perfectly for many set of parameters and gives almost perfect accuracies -for this specific data set. In any case, handling such issues is beyond the scope of this project.


## Previous & Gained Experience

In general I would say I have had a decent experience in coding. More specifically, I have taken a couple of basic programming courses during my undergraduate studies. I also had to code for other related courses like Artificial Intelligence, Theoretical Machine Learning, Analysic of Algorithms etc (however, all these courses were heavily theoretical and required only a bit of coding in order to compare results or see in practice how things work). 

Although I have seen a fair amount of Artificial Intelligence and Machine Learning, it happened that I have never seen neural networks until a couple of months ago. Back then I decided to take some mini-courses on neural networks and deep learning on the data camp platform (as part of the Data Science course of the CogMaster). It was an informative experience and I had finally the chance to explore the basic notions in neural networks. However, these courses have limited depth, they usually avoid the details and most of the times, for tasks and notions that are relative complex (like the ones in deep learning) they mostly teach one how to use the standard libraries. Although this is a useful skill to have, it leads one to treat many concepts as black boxes, thereby depriving one of the opportunity to know exactly what is going on. 

Having said that, I decided to do this project in order to obtain a deeper intuition and understanding on deep learning concepts. Indeed, I had for the first time the opportunity to explore in detail how backpropagation works. More specifically, given my choice to use a matrix approach combined with Batch Gradient Descent (instead of Stochastic Gradient Descent, which would have made things a bit easier), I had to write down all the matrix computations involved in the backpropagation algorithm (for example, I had to ensure that the dimensions of the matrices match, I had to examine whether I should use the original or the transpose matrix, to take the averages over all the training set when needed etc). In the more mathematical side, I went through the details of the use of the chain rule, in order to compute successively the partial derivaties. I also examined which activation function I should choose, encountering issues like sigmoid function being related to the vanishing gradient problem (and for this reason I chose the ReLU function for the hidden layers). Finally, I understood the importance of feature scaling and how it affects in practice the computations. All in all it was a very informative experience that will enable me to have a deeper understanding of the deep learning concepts, even when I use the standard libraries. 




