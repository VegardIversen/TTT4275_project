import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from keras.datasets import mnist

#Loading the MNIST dataset from keras
(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

# for i in range(9):
#     print(train_y[i])

# for i in range(9):  
#     plt.subplot(330 + 1 + i)
#     plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))

# plt.show()