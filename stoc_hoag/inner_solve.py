# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:49:53 2023

@author: ichel
"""
from numba import jit
import numpy as np
from helpers import fmu

class InnerSolver:
    def __init__(self, A, y, ATA, ATy):
        self.A = A
        self.samples = A.shape[0]
        self.features = A.shape[1]
        self.y = y
        self.ATA = ATA    #A^T * A, kept for avoiding expensive computations.
        self.ATy = ATy    #A^T * y, kept for avoiding expensive computations.


    def fake_inner(self, eps, mu, mean, sig):
        features = self.features
        ATA = self.ATA
        ATy = self.ATy
        x_opt = np.linalg.solve(ATA + mu * np.eye(features), ATy)
        epsilon = np.random.normal(mean, sig, features)
        return x_opt + epsilon
    
    @jit
    def sgd_solve_inner(self, x0,eps,mu,tmax,B,L):
        A = self.A
        y = self.y
        features = self.features
        samples = self.samples
        ATA = self.ATA
        ATy = self.ATy
        # initialise
        x_fin = x0
        # exact solution
        x_opt = np.linalg.solve(ATA + mu * np.eye(features), ATy)
        #
        #counter = 0
        for t in range(0,tmax):
            # set learning rate 
            alpha = 1.0/(L*((t+1.0))**0.5)
            # choose sample
            p = np.random.randint(0,samples,size = B)
            As = A[p,:]
            ys = y[p]
            # approximate gradient
            gradF = np.real((samples/B)*As.T @ (As @ x_fin - ys) + (mu)*x_fin)
            # update
            x_fin = x_fin - alpha * gradF
            # 
            if (np.linalg.norm(x_fin - x_opt,2) < eps):
                break
        error=np.linalg.norm(x_fin - x_opt,2)
        if (error>eps):
            print("Inner iteration fails to converge within "+ str(tmax) + " iterations")
            print("eps = ", eps)
            print("error = ",error)
            print("stepsize, alpha")
            assert(False)
        else:
            print("Passed inner iteration ", t, "iterations; error=", error, ", tolerance= ", eps, "alpha=",alpha)
        return x_fin
###################################################


    def solve_inner(self,lam,x0,eps,params):
        A = self.A
        y = self.y
        features = self.features
        samples = self.samples
        ATA = self.ATA
        ATy = self.ATy
    
        tmax=params["max_inner_its"]
        stoch=params["SGD_flag"] # GD or SGD
        B=params["SGD_batch_size"]
        assert(B>0)
        note_fn = []
        # Lipschit constant
        mu=float(fmu(lam))
        L = max(np.abs(np.linalg.eig(ATA+mu*np.eye(features))[0]))

        # initialise
        x_fin = x0
        # exact solution
        x_opt = np.linalg.solve(ATA + mu * np.eye(features), ATy)
        #
        counter = 0
        for t in range(0,tmax):
            counter += 1
            # set learning rate
            alpha = np.real(params["sgd_lr"](counter,L))
            assert(alpha>0)
            #
            if (stoch == True):
                # choose sample
                p = np.random.randint(0,samples,size = B)
                As = A[p,:]
                ys = y[p]
                # approximate gradient
                gradF = np.real((samples/B)*As.T @ (As @ x_fin - ys) + (mu)*x_fin)
            else:
                gradF = ATA @ x_fin - ATy + (mu)*x_fin
            #
            x_fin = x_fin - alpha * gradF
            #
            if "record_sgd" in params:
                #note_xes.append(x_fin)
                Axy = A@x_fin - y
                fnval = np.dot(Axy,Axy)/2 + (mu/2)*np.dot(x_fin,x_fin)
                note_fn.append(fnval)
            if (np.linalg.norm(x_fin - x_opt,2) < eps):
                break
        error=np.linalg.norm(x_fin - x_opt,2)
        if (error>eps):
            print("Inner iteration fails to converge within "+ str(tmax) + " iterations")
            print("eps = ", eps)
            print("error = ",error)
            assert(False)
        if params["report"]:
            print("Inner error=",error, "x=",x_fin)
    
        #print("Finished inner problem in " + str(counter) + " iterations")

        return x_fin, note_fn
