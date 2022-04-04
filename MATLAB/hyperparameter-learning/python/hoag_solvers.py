#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:45:43 2022

@author: cmhsp20
"""

from scipy.special import expit
import numpy as np
from scipy.sparse import linalg
from scipy import  optimize
from scipy.sparse.linalg import LinearOperator



        
class BaseSolver:
    def __init__(self, A, b,method = 'CG',maxiter=100):
        self.A = A
        self.b = b
        self.n = A.shape[1]
        self.method = method
        self.maxiter = maxiter
    
    def funcval(self,x,alpha):
        raise NotImplementedError('subclasses must override funcval!')
    
    def grad_x(self,x,alpha):
        raise NotImplementedError('subclasses must override grad_x!')
    
    def hess_x(self,x,alpha,w):
        raise NotImplementedError('subclasses must override hess_x!')
    
    def minimise_x(self,alpha,x0 = []):
        if not np.any(x0):
            x0 = np.random.randn(self.n,) 
        f = lambda x: self.funcval(x,alpha)
        grad = lambda x: self.grad_x(x, alpha)
        R = optimize.minimize(f, x0, method=self.method, jac=grad, options = { "maxiter" : self.maxiter})
        x = R.x
        
        return x


class Ridge_uv(BaseSolver):
    def __init__(self, A, b,method = 'CG',maxiter=100):
        super().__init__(A,b,method,maxiter)
        self.n = A.shape[1]*2
    
        
    def funcval(self,x,alpha):
        n = self.n//2
        u,v = x[:n], x[n:]
        res = self.A@(u*v) - self.b
        fval = np.dot(res,res)*.5 + np.dot(x,x)*alpha*.5
        return fval
    
    
    def grad_x(self,x,alpha):
        n = self.n//2
        u,v = x[:n], x[n:]
        g = self.A.T @ (self.A@(u*v)-self.b)
        gradx =  np.concatenate((v,u))*np.concatenate((g,g)) +alpha*x
        return gradx 
    
    def hess_x(self,x,alpha,w):
        n = self.n//2
        A = self.A
        u,v = x[:n], x[n:]
        w1,w2 = w[:n], w[n:]
        r = A.T@( A@(u*v)-self.b )
        Avw1 =  A.T@(A@(v*w1))
        Auw2 = A.T@(A@(u*w2))
        duu =  v* Avw1
        dvv =  u* Auw2
        dudv =  r*w2 + v*Auw2
        dvdu =  r*w1 + u*Avw1
                
        return  np.concatenate((duu+dudv, dvv+dvdu)) + alpha*w
    
    

class Ridge(BaseSolver):

    def __init__(self, A, b,method = 'CG',maxiter=100):
        super().__init__(A,b,method,maxiter)
        
    def funcval(self,x,alpha):
        res = self.A@x - self.b
        fval = np.dot(res,res)*.5 + np.dot(x,x)*alpha*.5
        return fval
    
    
    def grad_x(self,x,alpha):
        gradx =  self.A.T @ (self.A@x-self.b) +alpha*x
        return gradx 
    
    def hess_x(self,x,alpha,v):
        return  self.A.T @ (self.A@v) +alpha*v
    
    
    def minimise_x(self,alpha,x0 = []):
        n = self.n
        if not np.any(x0):
            x0 = np.random.randn(n,) 
        
        Atb = self.A.T@self.b
        M = LinearOperator((n,n), matvec= lambda x: self.A.T@(self.A@x)+alpha*x )
        res = linalg.cgs(M, Atb, maxiter=self.maxiter, tol = 1e-8)
        return res[0]


    
class Logistic(BaseSolver):

    def __init__(self, A, b,method = 'CG',maxiter=100):
        super().__init__(A,b,method,maxiter)
        
    def __sumlogexp(self,x):
    #implement sum(log(1+exp(-x))) to prevent numerical overflow
        xneg = np.minimum(x,0)
        xpos =  np.maximum(x,0)   
        s = np.sum( -xneg + np.log( (np.exp(xneg) +np.exp(-xpos)) ))
        return s
        
    
    def funcval(self,x,alpha):
        fval = self.__sumlogexp(self.b*(self.A@x)) +np.dot(x,x)*alpha*0.5
        return fval
    
    
    def grad_x(self,x,alpha):
        A = self.A
        gradx =  A.T@(-self.b*expit(-self.b*(A@x)))+alpha*x
        return gradx 
    
    #define the hessian at (x,y) applied to vector v
    def hess_x(self,x,alpha,v):
        w = self.b*(self.A@x)
        #compute  z = exp(w)/(1+exp(w))**2
        z = expit(w)*expit(-w)        

        return self.A.T@(z*(self.A@v))+alpha*v
        
    
    
class Logistic_uv(BaseSolver):
    def __init__(self, A, b,method = 'CG',maxiter=100):
        super().__init__(A,b,method,maxiter)
        self.n = A.shape[1]*2
                
    def __sumlogexp(self,x):
    #implement sum(log(1+exp(-x))) to prevent numerical overflow
        xneg = np.minimum(x,0)
        xpos =  np.maximum(x,0)   
        s = np.sum( -xneg + np.log( (np.exp(xneg) +np.exp(-xpos)) ))
        return s
    
    def funcval(self,x,alpha):
        n = self.n//2
        fval = self.__sumlogexp(self.b*(self.A@ (x[:n]*x[n:]))) +np.dot(x,x)*alpha*0.5
        return fval
    
    
    def grad_x(self,x,alpha):  
        n = self.n//2
        u,v = x[:n], x[n:]
        g = self.A.T@(-self.b*expit(-self.b*(self.A@(u*v))))
        gradx =  np.concatenate((v,u))*np.concatenate((g,g)) +alpha*x
        return gradx 
    
    #define the hessian at (x,y) applied to vector w
    def hess_x(self,x,alpha,w):   
        n = self.n//2
        u,v = x[:n], x[n:]
        
        h = self.b*(self.A@ (u*v))
        #compute  z = exp(h)/(1+exp(h))**2
        z = expit(h)*expit(-h)   
        r = self.A.T@(-self.b*expit(-h))
        
        
        w1,w2 = w[:n], w[n:]
        
        Avw1 =  self.A.T@( z*(self.A@(v*w1)) )
        Auw2 = self.A.T@( z*(self.A@(u*w2)) )
        duu =  v* Avw1
        dvv =  u* Auw2
        dudv =  r*w2 + v*Auw2
        dvdu =  r*w1 + u*Avw1
                
        return  np.concatenate((duu+dudv, dvv+dvdu)) + alpha*w
        
    
    
    
    