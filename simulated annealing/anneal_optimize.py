import numpy as np
from scipy.optimize import dual_annealing

def func(x):
    global teta
    T, C = x
    return T**2 + 2*C**2 + 0.5*T*C + 10*T + 20*C



lw = [50 , 0.5] 
up = [150, 2 ] 
global teta
teta = 1
sol = dual_annealing(func, bounds=list(zip(lw, up)))

print(sol.fun)