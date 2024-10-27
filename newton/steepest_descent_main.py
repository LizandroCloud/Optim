# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 05:04:58 2024

@author: lizst
"""

# -*- coding: utf-8 -*-
"""
Universidade Federal Fluminense
Departamento de Engenharia Química e de Petróleo
Lizandro de Sousa Santos
E-mail: lizandrosousa@id.uff.br
"""

"""
Seepest Descent Algorithm

x0-> estimativa inicial
n_derive1 -> primeira derivada da função
derive2 -> segunda derivada da função
alpha -> passo da otimização (learning rate)
maxiter -> número máximo de iterações
tol -> tolerância para convergência    

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math as m


# Função objetivo
def func(x):
    f =  x[0]**2 + x[1]**2
    return f


# Primeira derivada
def n_derive1(x):

    x_base = x.copy()

    for i in range(np.size(x)):
        xu = x_base.copy()
        xd = x_base.copy()
        xu[i] = x_base[i] + hi
        xd[i] = x_base[i] - hi
        grad_x[i] = (func(xu) - func(xd) )/(2*hi)

    grad = np.array(grad_x)
    return grad


def newtonM(x, alpha, maxiter, tol):
    x1=x[0]
    x2=x[1]
    erro = 1e3
    i = 0
    devx=1e3
    
    while  (devx>tol) or (erro>tol) or i>maxiter:
        
        g = n_derive1(x)
        x_novo = x - alpha*g 
        erro = abs (func( x_novo ) - func( x ) )
        devx = np.linalg.norm(x_novo-x)

        if i>maxiter:
            print('número máximo de iterações')
            break
        if x[0]>1e5 or x[1] >1e5:
            print('saindo por divergência')
            break

        i=i+1

        print("Iteração", i)
        print("erro=", erro)
        print("devx=", devx)
        x = x_novo
    return x_novo # no final retorna o último valor

nD = 2 # Número de dimensões
grad_x = np.zeros(nD)
xu = np.zeros(nD)
xd = np.zeros(nD)
hh = np.zeros([nD,nD])
x0 = np.array([0.2,0.2])
alpha = 0.2
hi= 1e-3

# Chamada da função:
    
x_opt = newtonM(x0, alpha, maxiter=50000, tol=1e-3)

print("Valor ótimo de x:", x_opt)
print("Valor ótimo da função", func(x_opt))