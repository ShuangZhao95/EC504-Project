# LSH application for EC504 project
#author: Fei Chen
#date: 4.40.2018
#group members: John Curci, Shuang Zhao
# import csv
import sys
import numpy as np
from scipy import stats


def LSH_hashing(Xt,r,b): #using the rxb combinations of hyperplanes to generate rxb binary coordinates of each point in the 'space'
    print('hashing')
    dim = len(Xt[0]);
    N = len(Xt)
    prod_matrix = np.zeros([N,b,r])
    hyp_plane = np.zeros([b,r,dim])
    for i in range(b):
        for j in range(r):
            hyp_plane[i,j] = np.array(np.random.uniform(low=-1.0, high=1.0, size=dim))
            for n in range(N):
                a1 = np.array([float(item) for item in hyp_plane[i,j]])
                a2 = np.array([float(item) for item in Xt[n]])
                if np.dot(a1, a2) > 0:
                    prod_matrix[n,i,j] = 1
                else:
                    prod_matrix[n,i,j] = 0
    return (hyp_plane ,prod_matrix)

def cluster(prod_matrix): #find the nearest neighbour set for each point
    print('clustering')
    (N, b, r) = prod_matrix.shape   
    # print(N)
    NN = dict()
    for i in range(N):
        NN[i] = []
        for j in range(N):
            if i==j:
                continue
            for m in range(b):
                # print(prod_matrix[j,m])
                # print(prod_matrix[i,m])
                if np.array_equal(prod_matrix[i,m], prod_matrix[j,m]):
                    NN[i].append(j)
                    # print('yes')
                    break               
    return NN
    
def purity(NN, Ytrain, Ytest): # compute the purity of clustering with given training lables and the classification accuracy of the test points
    N = len(NN)
    result = []
    Ytrain = np.array(Ytrain)
    Ytest = np.array(Ytest)
    for k, l in enumerate(NN):
        item = NN[l]
        item = [i for i in item if i < Ntrain]
        if item == []:
            result.append(' ')
            continue
        # print(item)
        l = np.array(item)
        l_class = Ytrain[l]
        cc = stats.mode(l_class).mode[0]
        result.append(cc)
    s = 0
    for i in range(Ntrain):
        if Ytrain[i]==result[i]:
            s += 1
    stest = 0
    for i in range(Ntrain, N):
        if Ytest[i - Ntrain]==result[i]:
            stest += 1
    print("the purity is " + str(s/Ntrain))
    print("the CCR is " + str(stest/Ntest))
    return (result[Ntrain:N],s/Ntrain,stest/Ntest)

def LSH(Xtrain, Xtest, Ytrain, Ytest, r, b): ## main function
    Xt = np.array(Xtrain + Xtest)
    dim = len(Xtrain[0])
    # hyp_plane_best = np.zeros([b,r,dim])
    (hyp_plane ,prod_matrix) = LSH_hashing(Xt, r, b)
    NN = cluster(prod_matrix)
    (Yhat,Purity,ccr) = purity(NN, Ytrain, Ytest)
    return (Yhat,Purity,ccr)
    
    
                
csvfilename1 = sys.argv[1] # train data
csvfilename2 = sys.argv[2] # train labels
csvfilename3 = sys.argv[3] # test data
csvfilename4 = sys.argv[4] # test labels
r = int(sys.argv[5])
b = int(sys.argv[6])
print('r= ' + str(r) + 'b= ' + str(b))
#
try :
    f1 = open(csvfilename1,'r') #training data
    f2 = open(csvfilename2,'r') #training labels
    f3 = open(csvfilename3,'r') #test data
    f4 = open(csvfilename4,'r') #test labels
except:
    print("file open failed")
    exit(1)
X_train = list(csv.reader(f1))
Y_train = list(csv.reader(f2))
X_test = list(csv.reader(f3))
Y_test = list(csv.reader(f4))
Ntrain = len(X_train)
Ntest = len(X_test)
# N = len(X_train)
D = len(X_train[0])

# r = 20
# b = 2

(Yhat,Purity,ccr) = LSH(X_train, X_test, Y_train, Y_test, r, b)
# f3.write(hp)
# print(ccr)
