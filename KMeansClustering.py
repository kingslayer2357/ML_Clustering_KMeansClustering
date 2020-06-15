# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:46:22 2020

@author: kingslayer
"""

#KMEANS CLUSTERING

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv(r"Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

#Using Elbow method to find optimum no. of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("ELBOW METHOD")
plt.xlabel("no. of clusters")
plt.ylabel("WCSS")
plt.show()


#fitting K-Means to X
kmeans=KMeans(n_clusters=5,init="k-means++",n_init=10,max_iter=300)
y_kmeans=kmeans.fit_predict(X)

#visulasing the clusters

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c="red",label="cluster 1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c="green",label="cluster 2")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c="blue",label="cluster 3")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c="orange",label="cluster 4")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c="violet",label="cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="yellow",label="centroids")
plt.legend()
plt.title("K-MEANS")
plt.xlabel("Income")
plt.ylabel('Score')
plt.show()