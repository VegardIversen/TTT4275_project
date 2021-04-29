import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from keras.datasets import mnist
import operator
import seaborn
import time
from sklearn.cluster import KMeans
from scipy.spatial import distance
import datetime

#Loading the MNIST dataset from keras
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# print('X_train: ' + str(train_X.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  '  + str(test_X.shape))
# print('Y_test:  '  + str(test_y.shape))

#  -----------------------------------------------------
# |                                                     |
# |                     Clustering                      |
# |                                                     |
#  -----------------------------------------------------

#Assuming we can use the function KMeans from the sklearn library as matlab-users are allowed to use the in-built kmeans function
def sortData(train_X, train_y):
    #Sorting data to each class, preparing for clustering

    numbCount = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    #Number count, keeps track of how many of each label y. 
    for i in range(len(train_y)):
        numbCount[train_y[i]] += 1

    sortedTrainY = np.argsort(train_y)  #sort after index
    sortedTrainX = np.empty_like(train_X)

    for i in range(len(train_y)):
        sortedTrainX[i] = train_X[sortedTrainY[i]]
    return sortedTrainX, numbCount
def cluster(train_X, train_y, M):
    #Flattening the 28x28 image as well
    clusterStart = time.time()
    sortedTrainX, numbCount = sortData(train_X, train_y)
    flattenedSortedTrainX = sortedTrainX.flatten().reshape(60000, 784)
    clusters = np.empty((len(numbCount), M, 784))
    before = 0
    after = 0
    for count, i in enumerate(numbCount):
        after += i
        clustered = KMeans(n_clusters=M, random_state=0).fit(flattenedSortedTrainX[before:after]).cluster_centers_
        before = after
        clusters[count] = clustered
        print(count)
    clusterEnd = time.time()
    #Returning a reshaped clustering
    return clusters.flatten().reshape(len(numbCount)*64, 784)

#  -----------------------------------------------------
# |                                                     |
# |              NN and KNN implementation              |
# |                                                     |
#  -----------------------------------------------------
class NN():
    def __init__(self, K=7):
        self.K = K
    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
    def predictCKNN(self, test_X, M):
        predictions = []
        fail_predictions = []
        success_predictions = []
        index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        start = time.time()
        clusters = cluster(self.train_X, self.train_y, M)
        clusterTimeEnd = time.time()
        print(f"Runtime of the clustering is: {str(datetime.timedelta(seconds = (time.time() - start)))}.")
        # for i in range(9):
        #     plt.subplot(330 + 1 + i)
        #     plt.imshow(clusters[510+i], cmap=plt.get_cmap('gray'))
        # plt.plot()
        # print('shape of clusters: ', np.shape(clusters))
        # print('range of cluster: ', len(clusters[0]))
        
        reshapedTestX = test_X.flatten().reshape((len(test_X), 784))
        # print('shape of reshapedTestX: ', np.shape(reshapedTestX))
        start = time.time()
        for i in range(len(test_X)):
            dist = []
            for count in range(len(clusters)):
                dist.append(distance.euclidean(reshapedTestX[i], clusters[count]))
            # print(dist)
            sortedDist = np.argsort(dist)[:self.K]

            classCount = {}
            for j in sortedDist:
                number = index[int(j//64)]
                #If the class is mentioned, +1
                if number in classCount:
                    classCount[number] += 1
                else: classCount[number] = 1
            # print(classCount)
            predictions.append(max(classCount, key=classCount.get))
        print(f"Runtime of the KNN with clustering is: {str(datetime.timedelta(seconds = (time.time() - start)))}.")
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
    def predictCNN(self, test_X, M):
        predictions = []
        fail_predictions = []
        success_predictions = []
        index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        start = time.time()
        clusters = cluster(self.train_X, self.train_y, M)
        clusterTimeEnd = time.time()
        print(f"Runtime of the clustering is: {str(datetime.timedelta(seconds = (time.time() - start)))}.")
        # for i in range(9):
        #     plt.subplot(330 + 1 + i)
        #     plt.imshow(clusters[510+i], cmap=plt.get_cmap('gray'))
        # plt.plot()
        print('shape of clusters: ', np.shape(clusters))
        print('range of cluster: ', len(clusters[0]))
        
        reshapedTestX = test_X.flatten().reshape((len(test_X), 784))
        print('shape of reshapedTestX: ', np.shape(reshapedTestX))
        start = time.time()
        for i in range(len(test_X)):
            dist = []
            for count in range(len(clusters)):
                dist.append(distance.euclidean(reshapedTestX[i], clusters[count]))
            # print(dist)
            NN_index = np.argmin(dist)
            # print(NN_index)
            if test_y[i] != index[int(NN_index//64)]:
                # fail_predictions.append([test_X[i], train_X[NN_index], test_y[i], train_y[NN_index]])
                fail_predictions.append([test_X[i], clusters[NN_index].flatten().reshape((28, 28))])
            else:
                # success_predictions.append([test_X[i], train_X[NN_index], test_y[i], train_y[NN_index]])
                success_predictions.append([test_X[i], clusters[NN_index].flatten().reshape((28, 28))])
            predictions.append(index[int(NN_index//64)])
        print(f"Runtime of the NN with clustering is: {str(datetime.timedelta(seconds = (time.time() - start)))}.")
        return predictions, success_predictions, fail_predictions

#  -----------------------------------------------------
# |                                                     |
# |                  Plot functions                     |
# |                                                     |
#  -----------------------------------------------------
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
def plotConfusionMatrix(confusion_matrix, testSize, trainingSize, text):
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
    plt.title(f'Confusion matrix for MNIST task\n Training size: {trainingSize}, Test size: {testSize} \n Error rate = {100 * error:.1f}% \n ')
    plt.savefig(f'./figures/Confusion_matrix_{text}_c{trainingSize}_t{testSize}_e{100*error:.0f}_raw.png', dpi=200)
    plt.show()
def plotFailedPredictions(fail_predictions, text):
    # for i in range(9):
    plt.subplot(330 + 1 )
    plt.title('Test')
    plt.imshow(fail_predictions[0][0], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 1 )
    plt.title('Predicted')
    plt.imshow(fail_predictions[0][1], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 2 )
    plt.title('Difference')
    plt.imshow(differenceImage(fail_predictions[0][0], fail_predictions[0][1]), cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 3)
    plt.imshow(fail_predictions[1][0], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 4)
    plt.imshow(fail_predictions[1][1], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 5)
    plt.imshow(differenceImage(fail_predictions[1][0], fail_predictions[1][1]), cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 6)
    plt.imshow(fail_predictions[2][0], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 7)
    plt.imshow(fail_predictions[2][1], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 8)
    plt.imshow(differenceImage(fail_predictions[2][0], fail_predictions[2][1]), cmap=plt.get_cmap('gray'))
    plt.savefig(f'./figures/{text}_failed_predictions.png', dpi=200)
    plt.show()
def plotSuccessPredictions(success_predictions, text):
    plt.subplot(330 + 1 )
    plt.title('Test')
    plt.imshow(success_predictions[0][0], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 1 )
    plt.title('Predicted')
    plt.imshow(success_predictions[0][1], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 2 )
    plt.title('Difference')
    plt.imshow(differenceImage(success_predictions[0][0], success_predictions[0][1]), cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 3)
    plt.imshow(success_predictions[1][0], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 4)
    plt.imshow(success_predictions[1][1], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 5)
    plt.imshow(differenceImage(success_predictions[1][0], success_predictions[1][1]), cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 6)
    plt.imshow(success_predictions[2][0], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 7)
    plt.imshow(success_predictions[2][1], cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 8)
    plt.imshow(differenceImage(success_predictions[2][0], success_predictions[2][1]), cmap=plt.get_cmap('gray'))
    plt.savefig(f'./figures/{text}_success_predictions.png', dpi=200)
    plt.show()

#  -----------------------------------------------------
# |                                                     |
# |                Distance functions                   |
# |                                                     |
#  -----------------------------------------------------
def differenceImage(img1, img2):
    a = img1-img2
    b = np.uint8(img1<img2) * 254 + 1
    return a * b
def eucledianDistance(img1, img2):
    return np.sum(differenceImage(img1, img2))
def eucDist(x1, x2):
    return np.linalg.norm(x1 - x2)

#  -----------------------------------------------------
# |                                                     |
# |                     Run code                        |
# |                                                     |
#  -----------------------------------------------------
def runNN(trainingSize, testSize, plotConfusionMat, plotFailedPred, plotSuccessPred):
    model = NN()
    model.fit(train_X[:trainingSize], train_y[:trainingSize])
    predictions, success_predictions, fail_predictions = model.predictNN(test_X[:testSize])
    if plotConfusionMat:
        plotConfusionMatrix(getConfusionMatrix(predictions), testSize, trainingSize, 'NN')
    if plotFailedPred:
        plotFailedPredictions(fail_predictions, 'NN')
    if plotSuccessPred:
        plotSuccessPredictions(success_predictions, 'NN')

def runCNN(trainingSize, testSize, M, plotConfusionMat, plotFailedPred, plotSuccessPred):
    model = NN()
    model.fit(train_X[:trainingSize], train_y[:trainingSize])
    predictions, success_predictions, fail_predictions = model.predictCNN(test_X[:testSize], M)
    if plotConfusionMat:
        plotConfusionMatrix(getConfusionMatrix(predictions), testSize, trainingSize, 'CNN')
    if plotFailedPred:
        plotFailedPredictions(fail_predictions, 'CNN')
    if plotSuccessPred:
        plotSuccessPredictions(success_predictions, 'CNN')

def runCKNN(trainingSize, testSize, M, plotConfusionMat, plotFailedPred, plotSuccessPred):
    model = NN()
    model.fit(train_X[:trainingSize], train_y[:trainingSize])
    predictions, success_predictions, fail_predictions = model.predictCKNN(test_X[:testSize], M)
    if plotConfusionMat:
        plotConfusionMatrix(getConfusionMatrix(predictions), testSize, trainingSize, 'CKNN')
    if plotFailedPred:
        plotFailedPredictions(fail_predictions, 'CKNN')
    if plotSuccessPred:
        plotSuccessPredictions(success_predictions, 'CKNN')

    # Load the data
    # Initialize the value of k
    # To getting the predicted class, iterate from 1 to the total number of training data points
    # Calculate the distance between test data and each row of training data. Here we will use Euclidean distance as our distance metric.
    # Sort the calculated distances in ascending order based on distance values
    # Get top k rows from the sorted array
    # Get the most frequent class of these rows
    # Return the predicted class

#  -----------------------------------------------------
# |                                                     |
# |                       Main                          |
# |                                                     |
#  -----------------------------------------------------
def main():
    # runNN(10000, 10000,True, False, False)
    # x, y = sortData(train_X, train_y)
    # clust = cluster(train_X, train_y, 64)
    # runNN(60000, 10000, True, False, False) #Takes 2 hours, best performance
    runCNN(60000, 10000, 64, True, True, True) #Takes 2-3 minutes, next best performance
    # runCKNN(60000, 10000, 64, True, False, False) #Takes 2-3 minutes, worst performance
    # print(np.shape(clust))
    return
if __name__=='__main__':
    main()
    