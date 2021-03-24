import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#maybe just import the dataset from sklearn, makes it much easier to work with. 
#------------Variables--------------#
classes = 3
iris_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
path = './iris.csv'
#-----------^Variables^-------------#

#---------Importing data------------#
def load_data(path, d=','):
    data = pd.read_csv(path, sep=',') #reading csv file, and splitting with ","
    #data.columns = ['sepal_length','sepal_width','petal_length','petal_width','species']#making columnnames, for easier understanding
    #data.describe()#this gives all the information you need: count, mean, std, min, 25%, 50%,75%,max
    return data
#--------^Importing data^-----------#

#----------Preprocessing------------#
data = load_data(path)

train, test = train_test_split(data, test_size=20/50, stratify= data["species"], random_state=42)
#---------^Preprocessing^-----------#

print(train)



