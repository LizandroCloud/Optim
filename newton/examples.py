"""
Universidade Federal Fluminense
Departamento de Engenharia Química e de Petróleo
Lizandro de Sousa Santos
E-mail: lizandrosousa@id.uff.br
"""

"""
Collection of basic functions for optimization
"""

import numpy as np

def uofpA(x):
    # Univariable objective function problem A
    f = x**2 -2*x # Objective function
    return f # Minimization

def van_de_vusse(x):
    # Objective Function for maximization of the concentration regarding van de Vusse Reactor
    k1 = 5/6 # kinetic constante
    k2 = 5/3 # kinetic constante
    k3 = 1/6 # kinetic constante
        
    cas1 = -(k1+x)/(2*k3) 
    cas2 = (((k1+x)**2 + 4*k3*x)**0.5)/(2*k3) 
    cas = cas1 + cas2 # Molar concentration of CA
    cbs = k1*cas/(x+k2) # Molar concentration of CB
    f = cbs # Objective function
    return -f # Maximum

def compressor_work(x):
    # Objective Function for minimization of the compression work of an ideal gas
    b = 0.287
    p1 = 1.0  #entrance pressure
    p2 = 1.3 # intermediate pressure
    p4 = 10.0 # intermediate pressure
    p3 = x # decision variables
    # P = (min(0,g1))**2 + (min(0,g2))**2
    f = ( (p2 / p1)**b + (p3 / p2)**b + (p4/p3)**b - 3 ) # Objective function
    return f
