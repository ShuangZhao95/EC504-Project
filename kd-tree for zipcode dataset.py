#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:22:43 2018

@author: zhaoshuang, John Curci
"""

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

### Brute-force NN implementation, uncomment to run time comparison

print('Begin brute-force solution.  Takes ~30 seconds for 1000 test points.')

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


###


### KD-tree Implementation of KNN

# Makes one node of a KD-tree.  Stores median value on node and remaining values in children.
# Recursively builds the rest of the tree until all there is one node for every data point.
def makeKDTree(data,dim,dim_iter=0):
    if len(data) == 1: # If this is a leaf we're done
        left = None
        right = None
        val = data[0]
        return (left, right, val)
    elif len(data) > 1: # Else recurse until leaf
        data.sort(key=lambda x: x[dim_iter])
        dim_iter = (dim_iter + 1) % dim
        half_index = int(len(data)/2)
        left = makeKDTree(data[:half_index],dim,dim_iter)       #Split data into sub-trees
        right = makeKDTree(data[half_index+1:], dim, dim_iter)
        val = data[half_index]
        return (left, right, val)


# Recursively navigates tree.  Finds K-Nearest Neighbors.  Returns K zip code results.

def getKNN(tree,xte,k,dim,neighbors=None,dim_iter=0):
    root = neighbors == None    # Set a flag if this is the root
    if root:                    # If this is the root, start the neighbors list
        neighbors = []
    if tree:                    # If the tree is not empty, run the function
        dist = sum([(xte[i]-tree[2][i])**2 for i in range(dim)])  # Leave as square distance (avoid unnecessary sqrt)
        boundary_dist = tree[2][dim_iter] - xte[dim_iter]         # Distance along current dimension from dividing plane
        if len(neighbors) < k:                                    # If list is not empty, add to list and sort
            neighbors.append((dist, tree[2]))
            neighbors.sort(key=lambda x: x[0])
        elif dist < neighbors[-1][0]:                             # If dist is smaller than largest element in list,
           neighbors[-1] = (dist, tree[2])                        # replace at end of list and re-sort
           neighbors.sort(key=lambda x: x[0])
        dim_iter = (dim_iter + 1) % dim                           # Cycle through vector dimensions
        getKNN(tree[boundary_dist<0], xte, k, dim, neighbors, dim_iter) # Navigate down tree
        if boundary_dist**2 < neighbors[0][0]:                          # Use square distance again
            getKNN(tree[boundary_dist>=0], xte, k, dim, neighbors, dim_iter) # If result could be outside of expected
                                                                             # region, check other sub-tree
    if root:                                                                   # If back to root, return result.
        return [elem[1] for elem in neighbors]

###


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
    if results[i][0][2][:3] == test_d[i][2][:3]:  # We only compare the first 3 digits of the zipcode
        t += 1
print('Accuracy for naiveNN: ', end = '')
print(t/1000.0)
print('Running time for kdtree: ', end = '')
print(time_for_kdtree)
