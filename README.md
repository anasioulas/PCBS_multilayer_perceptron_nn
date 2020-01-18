Two-layer perceptron neural network from scratch
===========================

The goal of my project was to create a multilayer (two layers specifically) perceptron neural network from scratch, i.e. without using the standard deep learning libraries of Python like keras or pytorch.

More specifically, I restricted myself to using only numpy for the construction of the neural network. For the sake of illustration, I trained and evaluated the performance my network on a specific data set (more details about that below). I used pandas and sklearn to prepare my data -the "from sratch" condition applied only to the construction of then neural network.

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Two-layer perceptron neural network from scratch](#two-layer-perceptron-neural-network-from-scratch)
    - [Preparation of the data](#preparation-of-the-data)
    - [The neural network](#the-neural-network)
    - [Previous & Gained Experience](#previous-&-gained-experience)

<!-- markdown-toc end -->


## Preparation of the data

I downloaded the Breast Cancer Wisconsin (Diagnostic) Data Set from <https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)>. More specifically, from 
<https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/> we downloaded the second file named "breast-cancer-wisconsin.data". We then converted it to a csv file, by renaming it as "breast-cancer-wisconsin.csv", so that our code can read it as a csv file. Then we created a script, namely `data_preparation.py`, used for the preparation of the data. 

THE CODE IS MISSING HERE




## The neural network

The script that constructs, trains and evaluates the neural network, namely [neural_network.py](neural_network.py), uses only numpy, as mentioned above. 

We used an object-oriented programming approach. Our script implements all the standard functionalities of such a neural network, like gradient descent (which incorporates forward and backward propagation). 

CHANGE MAYBE Although large parts of this script can used for other etc

Since our data set concerns a binary classification problem, we used the standard `cross-entropy loss` function during the training phase.

Also, we used the `Rectified Linear Unit (ReLU)` as the activation function for the hidden layers (in our specific example, we have only one intermediate layer) and the `sigmoid` function as the activation function for the output layer.

Since we used a small data set, we did not use mini-batches, but instead we run forward and backward propagation on the whole training set (`batch gradient descent`).

Finally, in order to classify the instances and eventually evaluate the performance of our neural network, we used a `threshold` of 0.5, which means that instances with output value greater than or equal to 0.5 are classified as 1 and instances with output value less than 0.5 are classified as 0.  



## Previous & Gained Experience

In general I would say I have had a decent experience in coding. More specifically, I have taken a couple of basic programming courses during my undergraduate studies. I also had to code for other related courses like Artificial Intelligence, Theoretical Machine Learning, Analysic of Algorithms etc (however, all these courses were heavily theoretical and required only a bit of coding in order to compare results or see in practice how things work). 

Although I have seen a fair amount of Artificial Intelligence and Machine Learning, it happened that I have never seen neural networks until a couple of months ago. Back then I decided to take some mini-courses on neural networks and deep learning on the data camp platform (as part of the Data Science course of the CogMaster). It was an informative experience and I had finally the chance to explore the basic notions in neural networks. However, these courses have limited depth, they usually avoid the details and most of the times, for tasks and notions that are relative complex (like the ones in deep learning) they mostly teach you to use the standard libraries. Although this is a useful skill to have, it leads you to treat many concepts as black boxes, thereby depriving you of the opportunity to know exactly what is going on. 

Having said that, I decided to do this project in order to obtain a deeper intuition and understanding on deep learning concepts. Indeed, I had for the first time the opportunity to explore in detail how backpropagation works. More specifically, given my choice to use a matrix approach combined with Batch Gradient Descent (instead of Stochastic Gradient Descent, which would have made things a bit easier), I had to write down all the matrix computations involved in the backpropagation algorithm (for example, I had to ensure that the dimensions of the matrices match, I had to examine whether I should use the original or the transpose matrix, to take the averages over all the training set when needed etc). I also examined which activation function I should choose, encountering issues like sigmoid function being related to vanishing gradient problem (and for this reason I chose the ReLU function for the hidden layers). Finally, I understood the importance of feature scaling and how it affects in practice the computations. 




