# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 22:38:29 2020

@author: Poojitha

"""
import numpy as np
import matplotlib.pyplot as plt


def Create_dataset():
    org_data =  np.genfromtxt(input("Enter the file name: "), skip_header = 1)
    np.random.seed(2)
    np.random.shuffle(org_data)
    trainhead = "136" + "\t" + "55"
    testhead = "34" + "\t" + "55"
    
    np.savetxt("Tarasani_Poojitha_Train.txt", org_data[0:136,:], fmt = '%s0', delimiter = "\t", newline = "\n", header = trainhead, comments = "")
    np.savetxt("Tarasani_Poojitha_Test.txt", org_data[136:170,:], fmt = '%s0', delimiter = "\t", newline = "\n", header = testhead, comments = "")
    return 0 

def Calculate_PY(X, W):
    Z = np.dot(X, W)
    PY = 1 / (1 + np.exp(-Z))
    return PY

def Cost(Y, PY):
    J = 0
    for i in range(Y.shape[0]):
        Cost = (Y[i]*np.log(PY[i], where = PY[i]>0)) + ((1-Y[i])* np.log(1-PY[i], where = (1 -PY[i]) >0))
        J = J + Cost
        J = - J/Y.shape[0]
    return J

def Gradient_Descent(alpha, X, Y, W, iterations):
    X = np.append(np.ones(Y.shape), X, axis = 1)
    J_values = np.zeros((iterations, 1))
    temp = np.zeros(Y.shape)
    for r in range(iterations):
        PY = Calculate_PY(X, W)
        for i in range(X.shape[1]):
            p=0
            for j in range(X.shape[0]):
                p = p + (PY[i]-Y[i])*X[j][i]
            temp[i] = W[i]-(alpha*p)/X.shape[0]
            W[i] = temp[i]
        J_values[r] = Cost(Y, PY)
    return W, J_values

Create_dataset()

Train = np.genfromtxt(input("Enter the training file name: "), skip_header = 1)
X_Train = Train[:,0:Train.shape[1] - 1]
Y_Train = np.reshape(Train[:,Train.shape[1] - 1],(Train.shape[0], 1))

iterations = 5000
W = np.random.rand(X_Train.shape[1]+1, 1)
print("weights =")
print(W)
Weights, J = Gradient_Descent(0.0001, X_Train, Y_Train, W, iterations)
print("weights =")
print(Weights)
print("Initial J")
print( J[0])
Pred = Calculate_PY(np.append(np.ones(Y_Train.shape), X_Train, axis = 1), Weights)
Final_Cost = Cost(Y_Train, Pred)
print("Final cost for Training set is: " + str(Final_Cost))

fig, ax = plt.subplots()
line = ax.plot(J, color="red", label='Alpha=0.0001')
ax.legend(loc='upper right')
plt.title("Iterations verses Cost (J)")
plt.xlabel('GD Iterations')
plt.ylabel('Cost (J)')
fig.savefig("Tarasani_Poojitha_P2.png")
plt.show()

Test = np.genfromtxt(input("Enter the Test file name: "), skip_header = 1)
Y_Test = np.reshape(Test[:,Test.shape[1] - 1],(Test.shape[0], 1))
X_Test = Test[:,0:Test.shape[1] - 1]

Pred = Calculate_PY(np.append(np.ones(Y_Test.shape), X_Test, axis = 1), Weights)
Final_Cost = Cost(Y_Test, Pred)
print("Final cost for Testing set is: " + str(Final_Cost))


TP = TN = FP = FN = 0
Pred = Pred > 0.5
for i in range(Y_Test.shape[0]):
    if Pred[i]:
        if Pred[i] == Y_Test[i]:
            TP = TP + 1
        else:
            FP = FP + 1
    elif Pred[i] == Y_Test[i]:
        TN = TN + 1
    else:
        FN = FN + 1

Accuracy = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1 = 2*(1/((1/Precision)+(1/Recall)))

print("True Positive TP = " + str(TP))
print("True Negative TN = " + str(TN))
print("False Positive FP = " + str(FP))
print("False Negative FN = " + str(FN))
print("Accuracy = " + str(Accuracy))
print("Precision = " + str(Precision))
print("Recall = "+ str(Recall))
print("F1 Score = " + str(F1))