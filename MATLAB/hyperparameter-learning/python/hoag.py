#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 22:11:53 2022

@author: cmhsp20

Script to test HOAG the gradient is implemented in two ways:
    by doing CGS with the Hessian
    by using SR1
"""

from hoag_solvers import Ridge,Logistic,Ridge_uv,Logistic_uv
import numpy as np
from scipy.sparse import linalg
import matplotlib.pyplot as plt

from libsvmdata import fetch_libsvm
from tqdm import tqdm

from scipy.sparse.linalg import LinearOperator 
import time

from scipy import  optimize


class bilevelsolver:
    
    
    
    def __init__(self,inner, outer):
        self.inner = inner
        self.outer = outer
        self.x = []
        self.current_cost = 0
        
    def __f_and_g(self,y,mode='SR1'):
        alpha = np.exp(y)
        xm = self.x
        x = self.inner.minimise_x(alpha,xm)
        dx_f = self.outer.grad_x(x,0)
        dyx_g = alpha*x
              
        
        if mode=='SR1':
            gamma=0.5 #parameter for SR1, set inside (0,1)
            s = xm - x
            dx_g = self.inner.grad_x(xm, alpha)
            t = np.dot(s,dx_g)/np.dot(dx_g,dx_g)
            u = s-gamma*t*dx_g       
            Bkg = gamma*t*dyx_g
            if np.dot(u,dx_g) > (1e-8 * np.sqrt(np.dot(dx_g,dx_g)*np.dot(u,u))):
                Bkg = Bkg + u* np.dot(u,dyx_g)/np.dot(u,dx_g)
       
        else: 
            M = LinearOperator((inner.n,inner.n), matvec= lambda v: inner.hess_x(x,alpha,v))
            grad = linalg.cgs(M, dyx_g, maxiter=100, tol = 1e-6)
            Bkg =  grad[0]            
            
        
        grad = -np.dot(dx_f,Bkg)
        
        fval = outer.funcval(x,0)
        self.current_cost = fval
        self.x = x
         

        return fval, grad
    
    

    
    def minimise(self,y,method = 'L-BFGS-B',mode='SR1',maxiter = 100):
        fvals = []
        def log_cost(x):
            fvals.append(self.current_cost)  
            
        #estimage the first step using finite differences
        x = self.inner.minimise_x(np.exp(y))
        self.x = x
        x1 = self.inner.minimise_x(np.exp(y+0.01))
        dx_f = self.outer.grad_x(x,0)
        grad =  -np.dot(dx_f,x1-x)       
        tau = .5/abs(grad)       
        y -= tau*grad
        
        fvals.append(self.outer.funcval(x,0))
        
        f_and_g = lambda y: self.__f_and_g(y,mode=mode)
        
        R = optimize.minimize(f_and_g, y, callback= log_cost, method=method, jac=True, options = { "maxiter" : maxiter, 'disp': True})        
        return R.x,fvals
    

    
    

def gradient_descent_bilevel(inner,outer,niter=100,mode='hess',yinit = 1,step=['adaptive']):
    
    if step[0]=='fixed':
        use_adaptive_stepsize = False
        tau = step[1]      
    elif step[0]=='adaptive':
        use_adaptive_stepsize = True
    else:
        use_adaptive_stepsize=False
        tau = step[1]
  
    y= yinit
    fvals = []
    gvals = []
    
    #estimate first descent direction by finite differences
    x = inner.minimise_x(np.exp(y))
    x1 = inner.minimise_x(np.exp(y+0.01))
    dx_f = outer.grad_x(x,0)
    grad =  -np.dot(dx_f,x1-x)
    
    if use_adaptive_stepsize:
        tau = .5/abs(grad)
    
    y -= tau*grad
    
    f = outer.funcval(x,0)
    
    for i in tqdm(range(niter)):
        fm = f
        xm = x
        
        alpha = np.exp(y)
        x = inner.minimise_x(alpha,x)
        dx_f = outer.grad_x(x,0)
        dyx_g = alpha*x
              
        
        if mode=='SR1':
            gamma=0.5 #parameter for SR1, set inside (0,1)
            s = xm - x
            dx_g = inner.grad_x(xm, alpha)
            t = np.dot(s,dx_g)/np.dot(dx_g,dx_g)
            u = s-gamma*t*dx_g       
            Bkg = gamma*t*dyx_g
            if np.dot(u,dx_g) > (1e-8 * np.sqrt(np.dot(dx_g,dx_g)*np.dot(u,u))):
                Bkg = Bkg + u* np.dot(u,dyx_g)/np.dot(u,dx_g)
       
        else: 
            M = LinearOperator((inner.n,inner.n), matvec= lambda v: inner.hess_x(x,alpha,v))
            grad = linalg.cgs(M, dyx_g, maxiter=100, tol = 1e-6)
            Bkg =  grad[0]
            
            
        
        grad = - np.dot(dx_f,Bkg)
        
        f = outer.funcval(x,0)
        if f>fm:
            use_adaptive_stepsize = False
            tau/=10
        if use_adaptive_stepsize:
            tau = .5/abs(grad)   
        
        if step[0]=='decay':
            tau = step[1]/(1+i)**0.8
        
        y = y - tau*grad
        if abs(grad)<1e-12:
            break
        
        
        
        fvals.append(f)
        gvals.append(np.abs(grad))
    
    return y, fvals, gvals
        
        
# %%

#define data matrix and vectors
mtype = 'logistic'
#mtype = 'ridge'
#mtype = 'logistic_l1'
#mtype = 'ridge_l1'

datatype = 'libsvm'
#datatype = 'random'


inner_iter=100
outer_iter=150 


if datatype == 'libsvm':
    names = ["australian_scale","a1a","a1a_test","covtype.binary","real-sim","news20.binary"]
    
    X,y = fetch_libsvm(names[-2])   
    
    m,n = X.shape
    
    #do a 90-10 split into training and test data
    p = np.random.permutation(m)
    S = m*9//10
    A = X[p[:S],:]
    b= y[p[:S]]
    A2 = X[p[S:],:]
    b2 = y[p[S:]]
else:
    m=1000
    n=400
    A = np.random.randn(m,n)
    b = np.random.randn(m,)

    A2 = A+ 0.5*np.random.randn(m,n)
    b2 = b+ 0.5*np.random.randn(m,)
if mtype == 'logistic':
    b2 -= b2.min()
    b2 /= b2.max()/2
    b2 -= 1
    
    b -= b.min()
    b /= b.max()/2
    b -= 1
print(mtype,datatype, A.shape)

#define inner and outer solvers

if mtype=='logistic':
    inner = Logistic(A,b,method='L-BFGS-B', maxiter = inner_iter)
    outer = Logistic(A2,b2)
    yinit = 1
elif mtype=='ridge':
    inner = Ridge(A,b,method='L-BFGS-B', maxiter = inner_iter)
    outer = Ridge(A2,b2)
    yinit = 1
elif mtype == 'ridge_l1':
    inner = Ridge_uv(A,b,method='L-BFGS-B', maxiter = inner_iter)
    outer = Ridge_uv(A2,b2)
    yinit = np.log(np.max(np.abs(A.T@b))/4)
elif mtype == 'logistic_l1':
    inner = Logistic_uv(A,b,method='L-BFGS-B', maxiter = inner_iter)
    outer = Logistic_uv(A2,b2)
    yinit = np.log(np.max(np.abs(A.T@b))/4)
else:
    print('method not defined')
    raise ValueError
    

# %% Solve Bilevel problem
'''
outer_iter=1000
step = ['decay',.1]

print('\n\nrunning bilevel with SR1\n')
start = time.time()
y2b,f2b,g2b = gradient_descent_bilevel(inner,outer,niter=outer_iter,mode='SR1',step=step)
end = time.time()
time2b = end-start



outer_iter=100
step = ['decay',0.1]
print('\n\nrunning bilevel with hessian\n')
start = time.time()
y1b,f1b,g1b = gradient_descent_bilevel(inner,outer,niter=outer_iter,mode='hess',step=step)
end = time.time()
time1 = end-start
print(y1b,y2b)

'''
BS = bilevelsolver(inner,outer)

#use the Hessian for gradient computation
start = time.time()
y1,f1 = BS.minimise(yinit,mode='hess',method='BFGS',maxiter=outer_iter)
x1 = inner.minimise_x(np.exp(y1))

end = time.time()
time1 = end-start

#use SR1 for gradient computation
start = time.time()
y2,f2 = BS.minimise(yinit,mode='SR1',method='BFGS',maxiter=outer_iter)

end = time.time()
time2 = end-start


# %% Cross validation curve

print('\n\nCross Validation...\n')

yvals = np.linspace(-10,10,50)
mCurve = []
for y in tqdm(yvals):
    alpha = np.exp(y)
    mCurve.append(outer.funcval( inner.minimise_x(alpha), 0 )  )

# %% display results
plt.plot([y1,y1], [np.min(mCurve), np.max(mCurve)],linewidth=4.0)
plt.plot([y2,y2], [np.min(mCurve), np.max(mCurve)],linewidth=2.0)
plt.plot(yvals,mCurve,linewidth=2.0) 

plt.show()


# %% Plot objective error against time
objmin = min(min(f1),min(f2))
plt.semilogy(np.linspace(0,time1,len(f1)), (f1-objmin)/objmin,linewidth=4, label='hess')
plt.semilogy(np.linspace(0,time2,len(f2)),(f2-objmin)/objmin, label='SR1')
plt.legend()
plt.show()

    
    