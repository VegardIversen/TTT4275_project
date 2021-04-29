import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from keras.datasets import mnist
import random
import operator

np.random.seed(3)

#Loading the MNIST dataset from keras
(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

# print(train_X[0])

def getTrainingSubset(subsetLength):
    #function: getTrainingSubset(subsetLength)
    #parameter 1: Size of the subset M 
    #return: [parameter 1] random elements from total set N 
    global train_X
    idx = np.random.randint(len(train_X[0]), size=subsetLength)
    return train_X[idx,:]

# t = getTrainingSubset(10)

def eucDist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN():
    def __init__(self, K=3):
        self.K = K
    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
    def predict(self, test_X):
        predictions = [] 
        for i in range(len(test_X)):
            #Iterate through every train_X and find the euclidian distance between test_X and train_X and put it in a np.array
            dist = np.array([eucDist(test_X[i], x_t) for x_t in self.train_X])
            #Sort the np.array in ascending order based on distance values
            sortedDist = dist.argsort()[:self.K]
            #Dictionary of how many times the class occurs in sortedDist
            classCount = {}
            for j in sortedDist:
                #If the class is mentioned, +1
                classCount[int(self.train_y[j])] += 1
            predictions.append(max(classCount.items(), key=operator.itemgetter(1))[0])
        return predictions



    # Load the data
    # Initialize the value of k
    # To getting the predicted class, iterate from 1 to the total number of training data points
    # Calculate the distance between test data and each row of training data. Here we will use Euclidean distance as our distance metric.
    # Sort the calculated distances in ascending order based on distance values
    # Get top k rows from the sorted array
    # Get the most frequent class of these rows
    # Return the predicted class

#Cool way of looping
# for i in range(9):
#     print(train_y[i])

#---------------V---PLOT---V---------------
# Plotting the first 9 MNIST dataset from keras
for i in range(9):  
    plt.subplot(330 + 1 + i)
    plt.imshow(t[i], cmap=plt.get_cmap('gray'))
plt.show()

#------------^PLOT^---------------