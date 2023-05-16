# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:07:22 2023

@author: ichel
"""

import numpy as np
import scipy
from scipy import optimize
from helpers import Nabla, fmu
from hessian import Neumann
from inner_solve import InnerSolver


class OuterSolver:
    def __init__(self, A, A_test, y, y_test, ATA, ATy, AAT):
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
        self.inner = InnerSolver(A, y, ATA, ATy)
        self.grads_o = Nabla(A, A_test, y, y_test, ATA, ATy)
        self.neu = Neumann(A, A_test, y, y_test, ATA, ATy)
            
    def bilvl_gs2(self): #CHECK
        ATA = self.ATA
        ATy = self.ATy
        A_test = self.A_test
        y_test = self.y_test
        features = self.features
    #
        def obj(lam):
            x_min_gs = np.linalg.solve(ATA + fmu(lam)* np.eye(features), ATy)
            Atxy_gs = A_test @ x_min_gs - y_test
            Cval_gs = np.dot(Atxy_gs,Atxy_gs)/2
            return Cval_gs 
    #
        result=scipy.optimize.minimize_scalar(obj,bounds=(0,1e3),tol=1e-12)
        #print(result)
        lam_min=result.x
        x_min_gs = np.linalg.solve(ATA + fmu(lam_min)* np.eye(features), ATy)
        Cvmin=obj(lam_min)
        #
        return fmu(lam_min), Cvmin, x_min_gs
    
    def HOAG_simplified_fix_inn(self, params2):
        ATA = self.ATA
        ATy = self.ATy
        A_test = self.A_test
        y_test = self.y_test
        features = self.features
        inner = self.inner
        grads_o = self.grads_o
        neu = self.neu
        
        params = params2.copy()
        #
        lam_start=params["lam_start"]
        iters=params["max_outer_its"]
        #
        mu=fmu(lam_start)
        L = max(np.linalg.eig(ATA+mu*np.eye(features))[0])
        #eta_alg = 1/(3*L)
        #
        eps_base=params["eps_base"]
        eps_decay=params["eps_decay_term"]
        note_C = []
        note_mu = []
        #Q = params["Q_for_vQ"]
        beta0 = params["learning_rate_outer"]
        beta = beta0
        #Starting lambda
        lam_alg = lam_start
        x_fin = np.zeros(features)
        #
        for j in range(2,iters):
            #print("Started iteration ", j)
            if params["exact_inner"]:
                # beware may be ill-posed if ATA singlular and mu small
                x_fin = np.linalg.solve(ATA + mu* np.eye(features), ATy)
            else:
                if params["eps_dec"]:
                    eps_cur = min(eps_base,1.0 / (j-1)**(0.5+eps_decay))
                else:
                    eps_cur = eps_base
                    #params.update({"max_inner_its":int(j)})
                if params["SGD_sim"]:
                    #set the mean/variance for noise. offset by 10 to remove instabilities at start
                    mean = params["sim_noise_mean"](j-1)
                    #print(mean)
                    x_fin = inner.fake_inner(eps_cur,fmu(lam_alg), mean, mean)
                elif "fast_inner" in params:  
                    # Lipschitz constant
                    L = max(np.abs(np.linalg.eig(ATA+fmu(lam_alg)*np.eye(features))[0]))
                    x_fin= inner.sgd_solve_inner(x_fin,eps_cur, fmu(lam_alg),params["max_inner_its"],params["SGD_batch_size"],L)
                else:
                    x_fin,tmp = inner.solve_inner(lam_alg,x_fin,eps_cur,params)
                
            #
            Atxy = A_test@x_fin - y_test
            Cval = np.dot(Atxy,Atxy)/2
            note_C.append(Cval)
            #
            if params["exact_vQ"]:
                nabxx2f = grads_o.get_nabxxF_dir(lam_alg)
                tmp= np.linalg.lstsq(nabxx2f, grads_o.get_nabxC_dir(x_fin),rcond=None)
                vQ=tmp[0]
            elif (params["neumannEuler"]):
                vQ = neu.get_vQ(x_fin,lam_alg,j,mu,params)
            else:
                vQ = neu.himplicitEuler(x_fin,lam_alg,j,mu,params)
            nablC = grads_o.get_nablC()
            nabxl2F = grads_o.get_nabxl2F(x_fin,lam_alg)
            nabhatL = nablC - np.dot(nabxl2F,vQ)
            if params["dec_outer"]:
                beta = beta0 * (1.0/(j-1)**0.5)
            lam_alg = lam_alg - beta * nabhatL
            mu=fmu(lam_alg)
            #print(mu)
            note_mu.append(mu)
        print("Finished after ",j," iterations")
        #plt.plot(tmp)
        return fmu(lam_alg), note_C, note_mu, nabhatL