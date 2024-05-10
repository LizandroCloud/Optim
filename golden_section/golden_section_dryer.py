# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:26:35 2024

@author: lizst
"""

import numpy as np

def fobj(T):
    
    
    A = 153.84
    Ce = 0.0253
    CL = 15
    Cp = 0.18
    CpA = 1045.75
    FA = 3.38e5
    dHC = 4.64e7
    P = 188
    T1 = 390
    U = 45
    Vt = 56
    Wd = 0.1765
    Wo = 0.5
    
    beta = (-67000 + (-0.2631125 + 0.0003*T)*Wo*np.exp(-0.2368125 + 0.0266*T) )
    # beta = np.exp2(T*0.01)
    
    C =   1e3 +1e7*( 1.767*np.log(Wo/Wd)/(beta*Vt) ) * ( (FA*CpA + U*A)*(T-T1)*Cp ) / ( dHC + P*Ce + CL ) 
    
    return C

def golden_section(xmin, xmax, tol, itemax):
    
    i = 0
    L= xmax - xmin
    x1 = 0.382*L
    x2 = (1-0.382) * L
    f1 = fobj(x1)
    f2 = fobj(x2)
    
    while abs(f1 - f2)>tol and i < itemax:
        # analisar intervalos
        i += 1
        print ("numero de iteracoes =", i)
        print ("menor f=", min(f1,f2))
        if f1>f2:
            xmin = x1
            L = (xmax - xmin)
            x1 = xmin + 0.382*L
            x2 = xmin + (1-0.382)*L
            print("x ótimo =", x2)
        elif f2>f1:
            xmax = x2
            L = (xmax - xmin)    
            x1 = xmin + 0.382*L
            x2 = xmin + (1-0.382)*L
            print("x ótimo =", x1)
        elif f1==f2:
            break
        
        f1 = fobj(x1)
        f2 = fobj(x2) 
    
    return x1, min(f1,f2)
        
        
xopt, fopt =  golden_section(xmin=390, xmax=700, tol=1e-15, itemax = 1e4)            

import matplotlib.pylab as plt
xx = np.linspace(390,700,100)
plt.plot(xx,fobj(xx))
# plt.plot(xopt,fopt,'o')
plt.show()
    
