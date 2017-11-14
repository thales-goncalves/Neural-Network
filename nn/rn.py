# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 13:52:07 2017

@author: Thales
"""

from sklearn.neural_network import MLPClassifier

import pandas as pd

#df = pd.read_csv('tae.txt', header=None)
#--
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data', header=None)

y_train = df.iloc[0:119,[5]].values
x_train = df.iloc[0:119 , [0,1,2,3,4]].values

y_test = df.iloc[120:150, [5]].values             
x_test = df.iloc[120:150, [0,1,2,3,4]].values                        



mlp = MLPClassifier(hidden_layer_sizes=(30), max_iter=20000, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-6, random_state=0,
                     learning_rate_init=.0001)

mlp.fit(x_train,y_train)

print("Training set score: %f" % mlp.score(x_train, y_train))
print("Test set score: %f" % mlp.score(x_test, y_test))


