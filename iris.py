import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#maybe just import the dataset from sklearn, makes it much easier to work with. 
#------------Variables--------------#
classes = 3
iris_names = ['Setosa', 'Versicolour', 'Virginica']
path = './Iris_TTT4275/iris.csv'
#-----------^Variables^-------------#

#---------Importing data------------#
def load_data(path, d=','):
    data = pd.read_csv(path, sep=',') #reading csv file, and splitting with ","
    data.columns = ['sepal_length','sepal_width','petal_length','petal_width','species']#making columnnames, for easier understanding
    #data.describe()#this gives all the information you need: count, mean, std, min, 25%, 50%,75%,max
    return data

#--------^Importing data^-----------#
data = load_data(path)
#---------Preprocessing-------------#
<<<<<<< HEAD
train, test = train_test_split(data, test_size = 0.4, stratify = data["species"], random_state = 42)# example for splitting dataset
#--------^Preprocessing^------------#

=======
train, test = train_test_split(data, test_size = 20/50, stratify = data['species'], random_state = 42)# example for splitting dataset
#--------^Preprocessing^------------#
print(train)
>>>>>>> e3f5095d4b0b993f9e2217ff0bf7833e6515af48
print(test)