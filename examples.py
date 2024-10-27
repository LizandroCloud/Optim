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

def van_de_vusse(x):
    # Objective Function for maximization of the concentration regarding van de Vusse Reactor
    k1 = 5/6 # kinetic constante
    k2 = 5/3 # kinetic constante
    k3 = 1/6 # kinetic constante
        
    cas1 = -(k1+x)/(2*k3) 
    cas2 = (((k1+x)**2 + 4*k3*x)**0.5)/(2*k3) 
    cas = cas1 + cas2 # Molar concentration of CA
    cbs = k1*cas/(x+k2) # Molar concentration of CB
    f1 = cbs # Objective function
    return -f1 # Maximum