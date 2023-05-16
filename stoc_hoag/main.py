# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:01:17 2023

@author: ichel
"""

import numpy as np
from inner_solve import InnerSolver
from outer_solve import OuterSolver
from setup import A, A_test, y, y_test, ATA, ATy, AAT
#
import matplotlib.pyplot as plt


##############################
###TESTS
##############################

solver = OuterSolver(A, A_test, y, y_test, ATA, ATy, AAT)

#Obtain exact solution
minmu, minC, x = solver.bilvl_gs2()
print("Optimal hyperparameter mu=",minmu, "\nOptimal objective function L=",minC)
print("Norm of x is ",np.linalg.norm(x))
print("Norm of Ax-y is ",np.linalg.norm(A@x-y))

print("Norm of A_t x-y_t is ",np.linalg.norm(A_test @x-y_test))
print("Norm of A is ",np.linalg.norm(A))



#Obtain solution using pseudo-sgd for the inner problem.

#number of iterations of outer loop
max_its = int(650)

#Parameters for HOAG
params_sgd={"lam_start":2,
        "max_outer_its":max_its,
        "learning_rate_outer":0.5,
        "dec_outer": False,
        # inner flags
        "exact_inner":False,
        "SGD_flag":True,
        "SGD_sim": True,
        "sim_noise_mean": lambda k: 0.05/k, #noise for simulated SGD solution error. set 1/k for strongly convex, 1/sqrt(k) otherwise.
        #"sim_noise_mean": lambda k: 0,
        "max_inner_its":int(1e6),
        "fast_inner":True,
        "eps_base":0.01,# tolerance for inner problem
        "eps_dec": True, #epsilon decaying T/F?
        "eps_decay_term":0.5, #If epsilon decaying then eps_t = eps_0/t^(0.5+edc)
        "sgd_lr": lambda k,L: (1/L)*  (1.0/(k+1)**(0.5)), # sgd learning rate for iterate k, Lipschitz const L
        "SGD_batch_size":10,
        # vQ params
        "exact_vQ": False,
        "neumannEuler": True,
        "vQ_learning_rate":0.1,# (aka, eta)
        "vQ_rate_dec": False, #whether learning rate for implicit euler is a decaying, square-summable sequence.
                             #by default, set eta = eta0 * n^(-gamma) for gamma in (0.5,1]
        "vQ_gamma": 0.5, #We set gamma for the v_Q learning rate.
        #"Q_for_vQ": lambda k, eta, mu: 30,  #Constant value for Q
        "Q_for_vQ": lambda k, eta, mu: (np.log(k) / ((-1)*np.log(1-eta*mu))),  #Increasing Q
        "C_x_samples":5,
        "C_l_samples":5,
        "F_xx_samples":5,
        #"F_xx_samples": lambda k: ,
        # other
        "report":False,
         }

runs = 1
opt_mu_runs = 0
gradient_runs = 0
feval_runs = np.zeros(max_its-2)
mus_runs = np.zeros(max_its-2)

for j in range(runs):
    print("Started run " + str(j) + " of " + str(runs))
    
    mu_from_alg, feval_alg, the_mus, gradient = solver.HOAG_simplified_fix_inn(params_sgd)
    print("Size of gradient of objective", gradient)
    print("Optimal mu (HOAG)=",mu_from_alg, ", reference=",minmu)
    print("Objective (HOAG)=",feval_alg[-1], ", reference=",minC)
    
    opt_mu_runs = opt_mu_runs + mu_from_alg
    gradient_runs = gradient_runs + gradient
    feval_runs = feval_runs + feval_alg
    mus_runs = mus_runs + the_mus
    
opt_mu_runs = opt_mu_runs / runs
gradient_runs = gradient_runs / runs
feval_runs = feval_runs / runs
mus_runs = mus_runs / runs

not_transient = 550
    
    
print("Optimal mu (HOAG)=",opt_mu_runs)
print("Objective (HOAG)=",feval_runs[-1])
lspace = np.array(range(1,max_its-1))
notes_curve = np.log(lspace) / np.sqrt(lspace)
sqrt_curve = 1 / np.sqrt(lspace)
linvspace = 1/(lspace)
linvspace2 = 1/(lspace**2)
tofit_y = np.log(np.abs(feval_runs[-not_transient:] - minC) / np.abs(minC))
tofit_x = np.log(lspace[-not_transient:])
z = np.polyfit(tofit_x, tofit_y, 1)
polz = np.poly1d(z)
raised_to_pow = np.exp(z[0]*np.log(lspace) + z[1])

plt.loglog(np.abs(feval_runs - minC) / np.abs(minC), label = 'HOAG')
plt.loglog(lspace,raised_to_pow, label = 'Fitted Line')

#plt.plot(tofit_x,polz(tofit_x), label = 'Fitted Log-Line')
#plt.loglog(polz(lspace), label = 'Fitted Line')

plt.loglog(linvspace, label = '1/t')
plt.loglog(sqrt_curve, label = '1/sqrt(t)')
plt.loglog(notes_curve, label = 'log(t)/sqrt(t)')
plt.legend()
tmp=plt.xlabel("Iterations")
tmp=plt.ylabel("$|L(\lambda) - L_\min| / |L_\min|$")
plt.savefig("a_run_of_STHOAG.jpg", bbox_inches ="tight")
print(mus_runs)
print("Rate of convergence (via fitted line)", z[0])