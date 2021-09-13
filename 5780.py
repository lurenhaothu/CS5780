#<GREDED>
import sys
import numpy as np
import numpy.matlib

#<GREDED>
from pylab import *
print('You\'re running python %s' % sys.version.split(' ')[0])

def l2distanceSlow(X, Z = None):
    if Z is None:
        Z = X

    n, d = X.shape
    m = Z.shape[0]
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            D[i, j] = 0.0
            for k in range(d):
                D[i, j] = D[i, j] + (X[i, k] - Z[j, k])**2
            D[i, j] = np.sqrt(D[i, j])
    return D

def innerproduct(X, Z = None):
    if Z is None:
        Z = X
    #raise NotImplementedError('Your code goes here!')
    G = np.matmul(X, np.transpose(Z))
    return G

def l2distance(X, Z = None):
    if Z is None:
        Z = X

    n, d1 = X.shape
    m, d2 = Z.shape
    assert(d1 == d2), "Dimensions must match!"
    S = np.transpose(np.matlib.repmat(np.diag(np.matmul(X, np.transpose(X))), m, 1))
    #print('\n', S.shape)
    R = np.matlib.repmat(np.diag(np.matmul(Z, np.transpose(Z))), n, 1)
    #print('\n', R.shape)
    G = innerproduct(X, Z)
    D2 = S + R - 2 * G
    D = np.sqrt(D2)
    return D

# Little test of the distance function
X1=rand(2,3);
print("The diagonal should be (more or less) all-zeros:", diag(l2distance(X1,X1)))
assert(all(diag(l2distance(X1,X1))<=1e-7))
print("You passed l2distance test #1.")

X2=rand(5,3);
Dslow=l2distanceSlow(X1,X2);
Dfast=l2distance(X1,X2);
print("The norm difference between the distance matrices should be very close to zero:",norm(Dslow-Dfast))
assert(norm(Dslow-Dfast)<1e-7)
print("You passed test #2.")

x1=np.array([[0,1]])
x2=np.array([[1,0]])
x1.shape
x2.shape
print("This distance between [0,1] and [1,0] should be about sqrt(2): ",l2distance(x1,x2)[0,0])
assert(norm(l2distance(x1,x2)[0,0]-sqrt(2))<1e-8)
print("You passed l2distance test #3.")
