# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:51:01 2023

@author: ichel
"""

"Helper functions for finding derivatives and other things"

import numpy as np

class Nabla:
    def __init__(self, A, A_test, y, y_test, ATA, ATy):
        self.A = A
        self.y = y
        self.samples = A.shape[0]
        self.features = A.shape[1]
        self.A_test = A_test
        self.y_test = y_test
        self.samples_t = A_test.shape[0]
        self.features_t = A_test.shape[1]
        self.ATA = ATA    #A^T * A, kept for avoiding expensive computations.
        self.ATy = ATy    #A^T * y, kept for avoiding expensive computations.
        

    #Get the (subsampled) Hessian of F
    def get_nabxxF(self, S,lam):
        samples = self.samples
        features = self.features
        A = self.A
        
        # S=sample batch size
        p = np.random.randint(0,samples,size = S)
        As = A[p,:]
        nabxxF_v = (samples/S)*(As.T @ As) + (fmu(lam)) * np.eye(features)
        return nabxxF_v

    #Get the (subsampled) Hessian of F, multiplied by predefined vector v for efficient computing
    def get_nabxxF_v(self, S,v,lam):
        samples = self.samples
        A = self.A
        
        # S=sample batch size
        p = np.random.randint(0,samples,size = S)
        As = A[p,:]
        nabxxF_v = (samples/S)*(As.T @ (As @ v)) + fmu(lam)*v
        return nabxxF_v
    
    #Get nabla_x C, evaluated at x.
    def get_nabxC(self, S,x):
        samples_t = self.samples_t
        A_test = self.A_test
        y_test = self.y_test
        
        p = np.random.randint(0,samples_t,size = S)
        As = A_test[p,:]
        ys = y_test[p]
        nabxC = (samples_t/S)*(As.T @ (As @ x - ys))
        return nabxC

    #Get nabla_{x \lambda} F, evaluated at x. Theory relies on this being bounded, need to check.
    def get_nabxl2F(self, x,lam):
        return dmu(lam)*x

    # Zero for this problem as outer problem independent of lambda
    def get_nablC(self):
        return 0

    # GD variants of above methods
    def get_nabxxF_dir(self,lam):
        features = self.features
        ATA = self.ATA
        nabxxF_v = ATA + fmu(lam)*np.eye(features)
        return nabxxF_v

    #Get the (subsampled) Hessian of F, multiplied by predefined vector v for efficient computing
    def get_nabxxF_v_dir(self,v,lam):
        ATA = self.ATA
        nabxxF_v = (ATA @ v) + fmu(lam)*v
        return nabxxF_v
    
    #Get \nabla_x C, evaluated at x.
    def get_nabxC_dir(self,x):
        A_test = self.A_test
        y_test = self.y_test
        nabxC = A_test.T @ (A_test @ x - y_test)
        return nabxC
        
# lower bound on fmu() enforced to avoid singular matrices
def fmu(lam):
    return np.exp(lam)+0.1
def dmu(lam):
    return np.exp(lam)