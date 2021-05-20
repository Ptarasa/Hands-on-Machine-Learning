# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:02:42 2020

@author: Poojitha Tarasani
"""

import numpy as np

def Create_dataset():
    org_data =  np.genfromtxt(input("Enter the file name: "), skip_header = 1)
    np.random.seed(5)
    np.random.shuffle(org_data)
    trainhead = "439" + "\t" + "11"
    validhead = "146" + "\t" + "11"
    testhead = "146" + "\t" + "11"
    
    sqr = np.power(org_data,3)
    data_split = np.hsplit(org_data, org_data.shape[1])
    sqr_split = np.hsplit(sqr, sqr.shape[1])
    data = np.concatenate((data_split[0],sqr_split[0]), axis = 1)
    for i in range(1, org_data.shape[1]-1):
        data = np.concatenate((data, data_split[i], sqr_split[i]), axis = 1)
    data = np.concatenate((data, data_split[org_data.shape[1]-1]), axis = 1)
    
    np.savetxt("Tarasani_Poojitha_Train.txt", data[0:439,:], fmt = '%s0', delimiter = "\t", newline = "\n", header = trainhead, comments = "")
    np.savetxt("Tarasani_Poojitha_Valid.txt", data[439:585,:], fmt = '%s0', delimiter = "\t", newline = "\n", header = validhead, comments = "")
    np.savetxt("Tarasani_Poojitha_Test.txt", data[585:731,:], fmt = '%s0', delimiter = "\t", newline = "\n", header = testhead, comments = "")
    return 0 

def regression(X, Y):
    X = np.append(np.ones(Y.shape), X, axis = 1)
    W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return W 

def cost(X, Y, W):
    X = np.append(np.ones(Y.shape), X, axis = 1)
    J = np.dot(np.ones(Y.shape).T, np.multiply(np.dot(X, W) - Y, np.dot(X, W) - Y))/(2 * X.shape[0])
    return J 

def rsquare(Cost, Y):
    denom = np.dot(np.ones(Y.shape).T, np.multiply(Y - np.mean(Y), Y - np.mean(Y)))/(2 * Y.shape[0])
    return 1 - (Cost / denom)

def adjrsquare(R_sqr, M, N):
    return 1 - (((1-R_sqr)*(M-1))/(M-N-1))

Create_dataset() 
Train =  np.genfromtxt("Tarasani_Poojitha_Train.txt", skip_header = 1)
Y_Train = np.reshape(Train[:,Train.shape[1] - 1],(Train.shape[0], 1))
X_Train = Train[:,0:Train.shape[1] - 1]
Weights = regression(X_Train, Y_Train)
TrainCost = cost(X_Train, Y_Train, Weights)
print("Final Model:")
print("\nThe Weights of the model are given below (n=) :")
print(Weights.T)
print("\nThe final cost for the Training set is J(W) = " + str(TrainCost))


Valid =  np.genfromtxt("Tarasani_Poojitha_Valid.txt", skip_header = 1)
Y_Valid = np.reshape(Valid[:,Valid.shape[1] - 1],(Valid.shape[0], 1))
X_Valid = Valid[:,0:Valid.shape[1] - 1]
ValidCost = cost(X_Valid, Y_Valid, Weights)
print("\nThe final cost for the Validation set is J(W) = " + str(ValidCost))



Test =  np.genfromtxt("Tarasani_Poojitha_Test.txt", skip_header = 1)
Y_Test = np.reshape(Test[:,Test.shape[1] - 1],(Test.shape[0], 1))
X_Test = Test[:,0:Test.shape[1] - 1]
TestCost = cost(X_Test, Y_Test, Weights)
R_Squared = rsquare(TestCost, Y_Test)
Adjusted_R = adjrsquare(R_Squared, X_Test.shape[0], X_Test.shape[1])
print("\nThe final cost for the Testing set is J(W) = " + str(TestCost))
print("\nThe final R-square for the Testing set is J(W) = " + str(R_Squared))
print("\nThe final Adjusted R-square for the Testing set is J(W) = " + str(Adjusted_R))
