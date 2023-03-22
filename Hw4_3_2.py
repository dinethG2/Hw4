from keras.datasets import mnist
import numpy as np
from scipy.special import softmax
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model
import torch

(train_X, train_y), (test_X, test_y) = mnist.load_data()
# print('X_train: ' + str(train_X.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  '  + str(test_X.shape))
# print('Y_test:  '  + str(test_y.shape))

# Sigmoid function
def sig(z):
 return 1/(1 + np.exp(-z))

# Set parameters
d1 = 300
d2 = 200
d3 = 100
k = 10
d = train_X.shape[0]


W1 = torch.rand(d1, 784)
W2 = torch.rand(d2, d1)
W3 = torch.rand(k,d2)

for i in range(len(train_X)):
    a_1 = sig(np.matmul(W1, np.reshape(train_X[i], (784))))
    a_2 = sig(np.matmul(W2,a_1))
    output = softmax(np.matmul(W3, a_2))
    # print(output)






