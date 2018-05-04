#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:15:09 2018

@author: zhaoshuang
"""

import scipy.io as sio
import numpy as np
import heapq
import time
data =  sio.loadmat('data_mnist_train.mat')
x_train = list(data['X_train'])
y_train = list(data['Y_train'])
data =  sio.loadmat('data_mnist_test.mat')
x_test = list(data['X_test'])
y_test = list(data['Y_test'])
# train size: 784*60000; test size: 784*10000
train_d = []
test_d = []
for i in range(60000):
    tt = list(x_train[i])
    tt.append(str(y_train[i][0]))
    train_d.append(tt)
for i in range(10000):
    tt = list(x_test[i])
    tt.append(str(y_test[i][0]))
    test_d.append(tt)
    
# kdtree
def makeKDTree(data,dim,dim_iter=0):
    if len(data) == 1:
        left = None
        right = None
        val = data[0]
        return (left, right, val)
    elif len(data) > 1:
        data.sort(key=lambda x: x[dim_iter])
        dim_iter += 1
        dim_iter = dim_iter % dim
        half_index = len(data) >> 1
        left = makeKDTree(data[:half_index],dim,dim_iter)
        right = makeKDTree(data[half_index+1:], dim, dim_iter)
        val = data[half_index]
        return (left, right, val)

def getKNN(tree,xte,k,dim,heap=None,dim_iter=0):
    root = heap==None
    if root:
        heap = []
    if tree:
        dist = sum([(xte[i]-tree[2][i])**2 for i in range(dim)])
        boundary_dist = tree[2][dim_iter] - xte[dim_iter]
        if len(heap) < k:
            heapq.heappush(heap, (-dist, tree[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, tree[2]))
        dim_iter += 1
        dim_iter = dim_iter % dim
        getKNN(tree[boundary_dist<0],xte,k,dim,heap,dim_iter)
        if boundary_dist**2 < -heap[0][0]:
            getKNN(tree[boundary_dist>=0], xte, k, dim, heap, dim_iter)
    if root:
        return [elem[1] for elem in heap]
  
try_test = test_d[0:10] + test_d[990:1000] + test_d[2190:2200] + test_d[3190:3200] + test_d[4190:4200] + test_d[5190:5200] + test_d[6190:6200] + test_d[7190:7200] + test_d[8190:8200] + test_d[9190:9200] 
# pick 100 test data(10 for each class)
one_test = test_d[0:1]
    
# Using kd-tree here
results = []
dim = 784
kd_tree = makeKDTree(train_d, dim)
t0 = time.time()
num = 0  # using num to monitor how many test point we have done
for t in one_test: # The test data can be changed here, we used tey_test to run like 3 hours.
    results.append(tuple(getKNN(kd_tree, t, 1, dim)))
#    print(num)     
    num += 1
t1 = time.time()
#print(results)
time_for_kdtree = t1 - t0
print('################################')
print('Test data label: ', end= '')
print(one_test[0][784])
print('Running time for kdtree: ', end = '')
print(time_for_kdtree)  
print('Nearest neighbor label: ', end= '')
print(results[0][0][784])
print('################################') 
      
# calculate accuracy below, it takes 3hours for 100 data point
'''
a = 0
for i in range(100):
    if results[i][0][784] == try_test[i][784]:
        a += 1
print('Accuracy for kdtree: ', end = '')
print(a/100)
'''  
    
# Using naive nn here
pred = []
t00 = time.time()
num = 0  # using num to monitor how many test point we have done
for i in one_test:  # The test data can be changed here, we used tey_test to run like 3 hours.
    nn = None
    s_dist = 10000000000000000
    for j in range(60000):
        dist = sum([(train_d[j][k] - i[k])**2 for k in range(dim)])
        if dist < s_dist:
            s_dist = dist
            nn = train_d[j][784]
    pred.append(nn)
    
    #print(num)
    num += 1
    
t11 = time.time()
time_for_naive = t11 - t00
print('################################')
print('Test data label: ', end= '')
print(one_test[0][784])
print('Running time for naive nn: ', end = '')
print(time_for_naive) 
print('Nearest neighbor label: ', end= '')
print(pred[0])
print('################################')

# calculate accuracy below, it takes 3hours for 100 data point
'''  
a = 0
for i in range(100):
    if pred[i] == try_test[i][784]:
        a += 1
print('Accuracy for naive nn: ', end = '')
print(a/100) 
'''   
    
    
    
    
    
    
    
    
    
