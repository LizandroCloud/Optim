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
def func(p2,p3):
    
    teta = 10
    b = 0.286
    p1 = 1 
    p4 = 10
    f = ( (p2 / p1)**b + (p3 / p2)**b + (p4/p3)**b - 3 )
    # print(P)
    return f


# Primeira derivada
def n_derive1(x1,x2):
        
    grad_x1 = (func(x1 + h, x2) - func(x1 - h, x2))/(2*h) # df_dx1
    grad_x2 = (func(x1, x2+h) - func(x1, x2-h))/(2*h)     #  df_dx2
    grad = np.array([grad_x1, grad_x2])                    # gradiente 
    
    return grad

def n_derive2(x1,x2):
 
    
    g_x0 = ( n_derive1(x1+h,x2) - n_derive1(x1-h,x2) ) / (2*h) # calculando a derivada segunda em relação a x1
    g_x1 = ( n_derive1(x1,x2+h) - n_derive1(x1,x2-h) ) / (2*h) # calculando a derivada segunda em relação a x2
         
     
    H = np.array([g_x0,g_x1])
    return H


def steepest(x, alpha, maxiter, tol):
    x1=x[0]
    x2=x[1]
    erro = 1e3 # valor inicial do erro
    i = 0
    devx=1e3

    while  (devx>tol) or (erro>tol):
        
        
        d = -n_derive1(x[0], x[1])    
        H = n_derive2(x[0], x[1])    
        alpha = d@d / (d@H@d)
        
        x_novo = x - alpha*n_derive1(x[0], x[1])       

        erro = abs (func( x_novo[0],x_novo[1] ) - func(x[0], x[1]) )
        devx = np.linalg.norm(x_novo-x)
        x = x_novo

        if i>maxiter:
            break
        if x[0]>1e5 or x[1] >1e5:
            print('saindo por divergência')
            break

        i=i+1

        print("Iteração", i)
        print("erro=", erro)
        print("devx=", devx)
    return x_novo # no final retorna o último valor

x0 = np.array([0.2,0.2])
alpha = 0.03
h = 1e-3
xg1=[]
xg2=[]

    
x_opt = steepest(x0, alpha, maxiter=5000, tol=1e-4)

print("Valor ótimo de x:", x_opt)
print("Valor ótimo da função", func(x_opt[0], x_opt[1]))
