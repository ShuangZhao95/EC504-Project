#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:22:43 2018

@author: zhaoshuang
"""

import heapq
import time
f1 = open('train_data.csv','r')
train_d = []
for line in f1:
    line = line.strip().split(',')
    for i in range(2):
        try:
            line[i] = float(line[i])
        except:
            pass
    train_d.append(line)
del(train_d[0])

f2 = open('test_data1000.csv','r')
test_d = []
for line in f2:
    line = line.strip().split(',')
    for i in range(2):
        try:
            line[i] = float(line[i])
        except:
            pass
    line[2] = line[2][3:8]
    test_d.append(line)
del(test_d[0])

pred = []   # prediction by the naive NN-method
start_time = time.time()
for i in range(1000):
    x1 = test_d[i][0]
    x2 = test_d[i][1]    
    d = 10000
    z = ''
    for j in range(len(train_d)):
        if ((x1 - train_d[j][0])**2 + (x2 - train_d[j][1])**2) < d:
            d = ((x1 - train_d[j][0])**2 + (x2 - train_d[j][1])**2)
            z = train_d[j][2]
    pred.append(z)
end_time = time.time()
time_for_naiveNN = end_time - start_time
t = 0
for i in range(1000):
    if pred[i][:3] == test_d[i][2][:3]:  # we only compare the first 3 number of zipcode
        t += 1
print('Accuracy for naiveNN: ', end = '')
print(t/1000.0)
print('Running time for naiveNN: ', end = '')
print(time_for_naiveNN)



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
results = []
dim = 2
kd_tree = makeKDTree(train_d, dim)
t0 = time.time()
for t in test_d:
        results.append(tuple(getKNN(kd_tree, t, 1, dim)))
t1 = time.time()
#print(results)
time_for_kdtree = t1 - t0
t = 0
for i in range(1000):
    if results[i][0][2][:3] == test_d[i][2][:3]:  # we only compare the first 3 number of zipcode
        t += 1
print('Accuracy for naiveNN: ', end = '')
print(t/1000.0)
print('Running time for kdtree: ', end = '')
print(time_for_kdtree) 
