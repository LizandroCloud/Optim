# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:26:35 2024

@author: lizst
"""
import math
import numpy as np

def fobj(x):
    f = 10 + x**2 - 10*np.cos(0.5*math.pi*x)
    return f

def golden_section(xmin, xmax, tol, itemax):
    
    i = 0
    L= xmax - xmin
    x1 = 0.382*L
    x2 = (1-0.382) * L
    f1 = fobj(x1)
    f2 = fobj(x2)
    
    while abs(f1 - f2)>tol and abs(x1-x2) and i < itemax:
        # analisar intervalos
        i += 1
        print ("numero de iteracoes =", i)
        print ("menor f=", min(f1,f2))
        if f1>f2:
            xmin = x1
            L = (xmax - xmin)
            x1 = xmin + 0.382*L
            x2 = xmin + (0.618)*L
            print("x ótimo =", x2)
        elif f2>f1:
            xmax = x2
            L = (xmax - xmin)    
            x1 = xmin + 0.382*L
            x2 = xmin + (0.618)*L
            print("x ótimo =", x1)
        elif f1==f2:
            break
        
        f1 = fobj(x1)
        f2 = fobj(x2) 
    
    return x2, f2
        
        
xopt, fopt =  golden_section(xmin=-5, xmax=5, tol=1e-15, itemax = 1e4)            

import matplotlib.pylab as plt
xx = np.linspace(-5.12,5.12,1000)
plt.plot(xx,fobj(xx))
plt.plot(xopt,fopt,'o')
plt.show()
    
