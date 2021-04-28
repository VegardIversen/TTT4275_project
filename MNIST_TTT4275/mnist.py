import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial import distance
from keras.datasets import mnist
import random
import operator
from operator import itemgetter
import seaborn
import time
from scipy.spatial import distance

start = time.time()

#Loading the MNIST dataset from keras
(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

# print(train_X[0])

# def chunkifySet(chunkSize):
#     #function: getTrainingSubset(subsetLength)
#     #parameter 1: Size of the subset M 
#     #return: [parameter 1] random elements from total set N 
#     global train_X
#     global train_y
#     return np.split(train_X, chunkSize), np.split(train_y, chunkSize)

# t = getTrainingSubset(10)
def differenceImage(img1, img2):
    a = img1-img2
    b = np.uint8(img1<img2) * 254 + 1
    return a * b

def eucledianDistance(img1, img2):
    return np.sum(differenceImage(img1, img2))

def eucDist(x1, x2):
    return np.linalg.norm(x1 - x2)

# def KNN(test_X, train_X, train_y, k):
#     dist = np.array([eucDist(test_X, x_t) for x_t in train_X])
#     sortedDist = dist.argsort()[:k]
#     print(sortedDist)
#     classCount = {}
#     for j in sortedDist:
#         if train_y[j] in classCount:
#             classCount[train_y[j]] += 1
#         else: classCount[train_y[j]] = 1
#     return classCount

# for i in range(10):
#     print(KNN(test_X[i], train_X[:1000], train_y, 10))


# print(test_y[:10])

class NN():
    def __init__(self, K=3):
        self.K = K
    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
    def predictKNN(self, test_X):
        predictions = [] 
        # print(test_X.shape[-1])
        for i in range(len(test_X)):
            #Iterate through every train_X and find the euclidian distance between test_X and train_X and put it in a np.array
            dist = np.array([distance.euclidean(test_X[i], x_t) for x_t in self.train_X])
            #Sort the np.array in ascending order based on distance values
            sortedDist = dist.argsort()[:self.K]
            # print(sortedDist)
            #Dictionary of how many times the class occurs in sortedDist
            classCount = {}
            for j in sortedDist:
                #If the class is mentioned, +1
                if self.train_y[j] in classCount:
                    classCount[self.train_y[j]] += 1
                else: classCount[self.train_y[j]] = 1
            # print(classCount)
            predictions.append(max(classCount, key=classCount.get))
        return predictions
    def predictNN(self, test_X):
        predictions = []
        fail_predictions = []
        success_predictions = []
        for i in range(len(test_X)):
            #Iterate through every train_X and find the euclidian distance between test_X and train_X and put it in a np.array
            dist = []
            for j in range(len(self.train_X)):
                dist.append(eucledianDistance(test_X[i], self.train_X[j]))
                # dist.append(eucDist(test_X[i], self.train_X[j]))
            # print(dist)
            NN_index = np.argmin(dist)
            if test_y[i] != train_y[NN_index]:
                # fail_predictions.append([test_X[i], train_X[NN_index], test_y[i], train_y[NN_index]])
                fail_predictions.append([test_X[i], train_X[NN_index]])
            else:
                # success_predictions.append([test_X[i], train_X[NN_index], test_y[i], train_y[NN_index]])
                success_predictions.append([test_X[i], train_X[NN_index]])
            # print(NN_index)
            # print(self.train_y[NN_index])
            predictions.append(self.train_y[NN_index])
        return predictions, success_predictions, fail_predictions

# k = 3 #K nearest neighbours
chunkSize = 10000
testSize = 100
model = NN()
model.fit(train_X[:chunkSize], train_y[:chunkSize])
print('train labels:', test_y[:100])
# model.fit(train_X[:10], train_y[:10])
# print(model.predict(test_X[:100]))
predictions, success_predictions, fail_predictions = model.predictNN(test_X[:testSize])
# print(test_X[0][16][18])

def getConfusionMatrix(predictions):
    confusion_matrix = np.zeros((10,10))
    for i, x in enumerate(predictions):
            confusion_matrix[test_y[i], x] += 1
    return confusion_matrix
def getConfusionMatrixNormalized(predictions):
    confusion_matrix = np.zeros((10,10))
    for i, x in enumerate(predictions):
            confusion_matrix[test_y[i], x] += 1
    return confusion_matrix/np.amax(confusion_matrix)

def plotConfusionMatrix(confusion_matrix):
    dia_sum = 0
    for i in range(len(confusion_matrix)):
        dia_sum += confusion_matrix[i, i]
        error = 1 - dia_sum / np.sum(confusion_matrix)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in class_names], columns = [i for i in class_names])
    plt.figure(figsize = (10,7))
    seaborn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion matrix for MNIST task\n Training size: {chunkSize}, Test size: {testSize} \n Error rate = {100 * error:.1f}% \n ')
    plt.savefig(f'./figures/Confusion_matrix_NN_c{chunkSize}_t{testSize}_e{100*error:.0f}_raw.png', dpi=200)
    plt.show()

# plotConfusionMatrix(getConfusionMatrix(predictions))
def plotFailPredictions(fail_predictions):
    # for i in range(9):
    for i in range(3):
        plt.subplot(330 + 1 + i)
        plt.imshow(fail_predictions[i][0], cmap=plt.get_cmap('gray'))
        plt.subplot(330 + 1 + 1 + i)
        plt.imshow(fail_predictions[i][1], cmap=plt.get_cmap('gray'))
        plt.subplot(330 + 1 + 2 + i)
        plt.imshow(differenceImage(fail_predictions[i][0], fail_predictions[i][1]), cmap=plt.get_cmap('gray'))
    plt.show()

# def plotSuccessPredictions(success_predictions):
#     for i in range(3):
#         plt.subplot(330 + 1 + i)
#         plt.imshow(test_X[i], cmap=plt.get_cmap('gray'))
#         plt.subplot(330 + 1 + 1 + i)
#         plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
#         plt.subplot(330 + 1 + 2 + i)
#         plt.imshow(differenceImage(test_X[i], train_X[i]), cmap=plt.get_cmap('gray'))
#     plt.show()

plotFailPredictions(fail_predictions)
# print(model.predict(test_X))
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

# #---------------V---PLOT---V---------------
# for i in range(9):  
#     plt.subplot(330 + 1 + i)
#     plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
# plt.show()

# for i in range(9):
#      plt.subplot(330 + 1 + i)
#      plt.imshow(test_X[i], cmap=plt.get_cmap('gray'))
# plt.show()
# for i in range(9):  
#      plt.subplot(330 + 1 + i)
#      plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
# plt.show()
# for i in range(9):  
#      plt.subplot(330 + 1 + i)
#      plt.imshow(train_X[i+9], cmap=plt.get_cmap('gray'))
# plt.show()
#------------^PLOT^---------------

# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {(end - start)/60} minutes.")