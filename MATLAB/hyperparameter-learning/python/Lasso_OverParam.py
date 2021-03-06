#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:11:19 2022

@author: cmhsp20

Solvers for doing
min_x alpha*|x|_1 + |Ax - y|^2/2
solved by 
min_{u,v} alpha*|u|^2/2 + alpha*|v|^2/2 + |A (uv) - y|^2/2

"""

from libsvmdata import fetch_libsvm

from scipy.special import expit
import numpy as np
from scipy import  optimize
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.sparse.linalg import LinearOperator,spsolve
from scipy.sparse import linalg, identity, diags

import time



class Lasso_uv:
    def __init__(self):
        self.fvals = []
        self.x = []
        self.y = []
        self.n = 0
        self.current_cost = 0
    
    
    def L(self,z):
        raise NotImplementedError('subclasses must override L!')
    
    def nabla_L(self,z):
        raise NotImplementedError('subclasses must override nabla_L!')
    
    def hess_L(self,z,v):
        raise NotImplementedError('subclasses must override hess_L!')
              
    def funcval(self,x,y):
        return self.L(x*y)+np.dot(x,x)*0.5+np.dot(y,y)*0.5
    
    def grad_x(self,x,y):
        return x+y*self.nabla_L(x*y)
    
    def hess_x(self,x,y,v):
        return v + y*self.hess_L(x*y,y*v)
    
    def grad_xy(self,x,y,v):
        return self.nabla_L(x*y)*v + y*self.hess_L(x*y,x*v)
    
    #solve the inner problem with y fixed
    def minimise_x(self,y,x0 = [],method = 'L-BFGS-B',maxiter = 100):
        if not np.any(x0):
            x0 = np.random.randn(self.n,) 
        f = lambda x: self.funcval(x,y)
        grad = lambda x: self.grad_x(x, y)
        R = optimize.minimize(f, x0, method=method, jac=grad, options = { "maxiter" : maxiter, 'disp': False})        
        return R.x
    
    #function value and gradient of the single level problem
    def __singlelevel_f_and_g(self,xy):
        n = len(xy)//2
        x = xy[:n]
        y = xy[n:]
        fval = self.funcval(x,y)
        grady = self.grad_x( y,x)
        gradx = self.grad_x( x,y)
        grad = np.concatenate((gradx,grady))
        self.current_cost = fval
        self.x = x
        self.y = y
        return fval,grad
        
    
    #function value and gradient of outer problem as a function of y, with inner problem over x
    def __bilevel_f_and_g(self,y,mode='',method='L-BFGS-B',maxiter=100):
        xm= self.x
        x = self.minimise_x(y,maxiter=maxiter,method=method)
        self.x = x
        self.y = y
        fval = self.funcval(x,y)
        
        self.current_cost = fval
        grad = self.grad_x(y,x)
                
        if mode=='SR1':
            dx_f = self.grad_x(x,y)
            gamma=0.5 #parameter for SR1, set inside (0,1)
            s = xm - x
            dx_g = self.grad_x(xm, y)
            t = np.dot(s,dx_g)/np.dot(dx_g,dx_g)
            u = s-gamma*t*dx_g       
            Bkg = gamma*t*dx_f
            if np.dot(u,dx_g) > (1e-8 * np.sqrt(np.dot(dx_g,dx_g)*np.dot(u,u))):
                Bkg = Bkg + u* np.dot(u,dx_f)/np.dot(u,dx_g)
       
            grad -= self.grad_xy(y,x, Bkg )
        elif mode=='hess': 
            dx_f = self.grad_x(x,y)
            M = LinearOperator((self.n,self.n), matvec= lambda v: self.hess_x(x,y,v))
            g = linalg.cgs(M, dx_f, maxiter=100, tol = 1e-6)
            grad -= self.grad_xy(y,x, g[0] )       

        return fval, grad
    
    
    def bilevel_solve(self,x=[],y=[],methods={'outer':'L-BFGS-B', 'inner': 'CG'},maxiter = {'outer':100, 'inner':10},mode=''):
        if not np.any(x):
            self.x = np.random.randn(self.n,)
        if not np.any(y):
            self.y = np.random.randn(self.n,)
            
        fvals = []
        def log_cost(x):
            fvals.append(self.current_cost)  
            
        innerfun = lambda y: self.__bilevel_f_and_g(y,mode,method = methods['inner'], maxiter=maxiter['inner'])
        R = optimize.minimize(innerfun, self.y, callback = log_cost, method=methods['outer'],jac=True, options = { "maxiter" : maxiter['outer']})
        return R.x,fvals
    
    def singlelevel_solve(self,x=[],y=[],method='L-BFGS-B',maxiter = 100):
        if not np.any(x):
            self.x = np.random.randn(self.n,)
        if not np.any(y):
            self.y = np.random.randn(self.n,)
        
        fvals = []
        def log_cost(x):
            fvals.append(self.current_cost)  
         
            
        xy = np.concatenate((self.x,self.y))
        R = optimize.minimize(self.__singlelevel_f_and_g, xy, callback = log_cost, method=method,jac=True, options = { "maxiter" : maxiter})
        return R.x, fvals
    
    
    

class Ridge_uv(Lasso_uv):
    def __init__(self, A, b,alpha):
        super().__init__()
        self.A = A
        self.b = b
        self.alpha = alpha
        self.n = A.shape[1]
        
        n = self.n
        m = self.A.shape[0]
        if n<=m:
            self.AtA = A.T@A
    
    def L(self,z):
        res = self.A@(z) - self.b
        return np.dot(res,res)*.5/self.alpha
    
    def nabla_L(self,z):
        return self.A.T@(self.A@z - self.b)/self.alpha
    
    def hess_L(self,z,v):
        return self.A.T@(self.A@v)/self.alpha
    
    def minimise_x(self, y,method = 'CG',maxiter = 20):
        
        n = self.n
        m = self.A.shape[0]
        if method=='direct': #exact solve
            if n<m:
                M = diags(y)*self.AtA @diags(y)+self.alpha*identity(self.n)
                return spsolve(M,y*(self.A.T@b) )
            else:
                M = self.A@ diags( y*y)@self.A.T+alpha*identity(m)
                a = spsolve(M,self.b)
                return y*(self.A.T@a)
        else:  #use conjugate gradient
            if m<n:
                M = LinearOperator((m,m), matvec= lambda x:  self.A@( y*y*(self.A.T@x ) )+alpha*x )
                res = linalg.cgs(M, self.b, maxiter=maxiter, tol = 1e-8)
                return y*(self.A.T@res[0])
            else:          
                Atb = self.A.T@self.b
                M = LinearOperator((n,n), matvec= lambda x:  y*( self.AtA@(x*y) )+alpha*x   )
                
                res = linalg.cgs(M, y*Atb, maxiter=maxiter, tol = 1e-8)
                return res[0]
   
class Logistic_uv(Lasso_uv):
    def __init__(self, A, b,alpha):
        super().__init__()
        self.A = A
        self.b = b
        self.alpha = alpha
        self.n = A.shape[1]
        
    def __sumlogexp(self,x):
    #implement sum(log(1+exp(-x))) to prevent numerical overflow
        xneg = np.minimum(x,0)
        xpos =  np.maximum(x,0)   
        s = np.sum( -xneg + np.log( (np.exp(xneg) +np.exp(-xpos)) ))
        return s
    
    def L(self,z):
        return self.__sumlogexp(self.b*(self.A@z)) /self.alpha
    
    def nabla_L(self,z):
        return self.A.T@(-self.b*expit(-self.b*(self.A@z)))/self.alpha
    
    def hess_L(self,z,v):    
        w = self.b*(self.A@z)
        #compute  z = exp(w)/(1+exp(w))**2
        z = expit(-w)*expit(w)     

        return self.A.T@(z*(self.A@v))/self.alpha





#??define problems
mtype = 'logistic'
datatype = 'libsvm'
#datatype = 'rand'
#mtype = 'ridge'
if datatype == 'libsvm':
    names = ["mnist","w8a","cadata","housing","abalone","australian_scale",
             "a8a","leukemia","a1a","a1a_test","covtype.binary",
             "real-sim","news20.binary"]   
    A,b = fetch_libsvm(names[0])       
    m,n = A.shape
 
else:
    m=300
    n=300
    A = np.random.randn(m,n)
    b = np.random.randn(m,)
    
alpha = np.max(np.abs(A.T@b))/100


if mtype == 'logistic':
    b -= b.min()
    b /= b.max()/2
    b -= 1
    X = Logistic_uv(A,b,alpha)   

else:
    X = Ridge_uv(A,b,alpha)

print(mtype,datatype, A.shape)


#doing a single level solve
xinit = np.random.randn(n,)
yinit = np.random.randn(n,)

start = time.time()
F,fvals1 = X.singlelevel_solve()
x1,y1 = X.x,X.y
end = time.time()
obj1 = np.min(fvals1)
time1 = end-start

#doing a bilevel solve

#assume that inner problem has been solved exactly
start = time.time()
maxiter = {'outer':100, 'inner':10}
methods = {'outer':'L-BFGS-B', 'inner': 'L-BFGS-B'}
F, fvals2 = X.bilevel_solve(mode='',maxiter = maxiter,methods = methods)
x2,y2 = X.x,X.y
obj2 = np.min(fvals2)
end = time.time()
time2 = end-start

#use implicit formula with SR1
start = time.time()
methods = {'outer':'L-BFGS-B', 'inner': 'L-BFGS-B'}
F, fvals3 = X.bilevel_solve(mode='sr1',maxiter = maxiter,methods = methods)
x3,y3 = X.x,X.y
obj3 = np.min(fvals3)
end = time.time()
time3 = end-start



#plot the results
objmin = min([obj1,obj2])
plt.semilogy(np.linspace(0,time1,len(fvals1)), fvals1 - objmin,label='single')
plt.semilogy(np.linspace(0,time2,len(fvals2)),  fvals2 - objmin,label='bilevel')

plt.semilogy(np.linspace(0,time3,len(fvals3)),  fvals3 - objmin,label='bilevel-sr1')

'''
objmin = np.minimum(obj1,obj2)
plt.semilogy(fvals1 - objmin,label='single')
plt.semilogy( fvals2 - objmin,label='bilevel')
plt.semilogy(fvals3 - objmin,label='bilevel-sr1')
'''
plt.legend()
plt.show()


