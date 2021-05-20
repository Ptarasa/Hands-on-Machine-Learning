# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 01:18:56 2020

@author: Poojitha Tarasani
"""

import numpy as np
import matplotlib.pyplot as plt

def dist(x1, y1, x2, y2):
    return np.sqrt(np.power((x1 - x2), 2) + np.power((y1 - y2), 2))

def assign(x, y, index):
    distance = dist(x, y, centroids[0][0], centroids[0][1])
    clusternum = 0
    for i in range(1, centroids.shape[0]):
        if distance >= dist(x, y, centroids[i][0], centroids[i][1]):
            distance = dist(x, y, centroids[i][0], centroids[i][1])
            clusternum = i
    labels[index][0] = x
    labels[index][1] = y
    labels[index][2] = clusternum
    labels[index][3] = centroids[clusternum][0] 
    labels[index][4] = centroids[clusternum][1]
    return 0

def plot_graph():
    plt.scatter(data[:,0], data[:,1])
    for point in centroids:
        plt.scatter(point[0], point[1], color='red', marker = "^")
    plt.show()
    
def plot_graph1():
    plt.scatter(data[:,0], data[:,1], c=labels[:,2])
    for point in centroids:
        plt.scatter(point[0], point[1], color='red', marker = "^")
    plt.show()

def kmeans():
    count = 0 
    for point in data:
        assign(point[0],point[1],count)
        count = count + 1
    for i in range(centroids.shape[0]):
        for point in labels:
            if i == point[2]:
                new_centroids[i][0] = new_centroids[i][0] + point[0]
                new_centroids[i][1] = new_centroids[i][1] + point[1]
                counts[i] = counts[i] + 1
        if counts[i] > 0:
            centroids[i][0] = new_centroids[i][0]/counts[i]
            centroids[i][0] = new_centroids[i][0]/counts[i]
        else:
            print("Cluster " + str(i) + " has no data-points.")
    return 0
    
def cost():
    cost = 0
    for point in labels:
        cost = cost + dist(point[0], point[1], point[3], point[4])
    return cost/labels.shape[0]
    
data =  np.genfromtxt(input("Enter the dataset file name: "), skip_header = 1)
centroids =  np.genfromtxt(input("Enter the Centroid file name: "), skip_header = 1)
#print the first centroids
print("The initial centroid coordinates are given below:")
print(centroids)

labels = np.zeros((data.shape[0], (2 * data.shape[1]) + 1))
new_centroids = np.zeros(centroids.shape)
counts = np.zeros((centroids.shape[0], 1))
print("The plot before clustering: ")
plot_graph()
kmeans()

for i in range(5):
    labels = np.zeros((data.shape[0], (2 * data.shape[1]) + 1))
    new_centroids = np.zeros(centroids.shape)
    counts = np.zeros((centroids.shape[0], 1))
    kmeans()

    

print("The plot of the cluster data.")    
plot_graph1()


print("The coordinates of the final centroids is given below:")
print(centroids)


print("overall error")
print(cost())
