# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:46:07 2023

@author: ichel
"""

import numpy as np
import matplotlib.pyplot as plt

# Dimensions of A
samples = 10 # rows
features = 20 # columns

#################
# Generate A
A = np.random.randn(samples,features)
if True:#rescale A
    A=A/np.linalg.norm(A)
#########
# Initialize some random sparse solution
x0 = np.zeros((features,1))
p = np.random.permutation(features)
print(p[:10])
x0[p[:10]] = np.random.randn(10,1)
x0 = np.reshape(x0,-1)
#######################
# Generate output y (with small noise perturbation)
y = A@x0+0.001*np.random.randn(samples)
###################
# plotting of solution x
plt.stem(x0)
plt.ylabel('$x$')
plt.xlabel('index')
plt.show()
#################
# Generate test date: matrix A_t and outputs y_t 
samples_t = 5 # size of test data
A_test = np.random.randn(samples_t,features)
#A_test=A_test/np.linalg.norm(A_test)
y_test = A_test@x0 
################
# Get A^T*A and A^T*y for future calculations
ATA = A.T @ A
ATy = A.T @ y
AAT = A @ A.T
##########################
# Get Lipschitz constants
mu_limax=1
Lbar = np.linalg.norm(ATA + (mu_limax)*np.eye(features))
#Find Limax, the greatest value of L_i over 1-sample evaluations of Hessian
Lis = []
for i in range(1,samples):
    Ai = A[i,:]
    nab_xx2F = samples*(Ai.T @ Ai) + (mu_limax)*np.eye(features)
    Lis.append(np.linalg.norm(nab_xx2F,2))
    
# Get Lipschitz constant for outer problem
M = np.linalg.norm(A_test.T @ A_test, 2)
#########
# plotting of Lipschit constants
plt.stem(Lis)
plt.ylabel('L')
plt.xlabel('index')
plt.show()
Limax = max(Lis)