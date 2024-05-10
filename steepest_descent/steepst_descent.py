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
def func(x1,x2):
    f = x1**2 + 25*x2**2
    return f

fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')

xa = np.linspace(-1.5,1.5,100)
xb = np.linspace(-0.3,0.3,100)
xx, yy = np.meshgrid(xa,xb)
zz = func(xx,yy)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Fobj')
ax.plot_surface(xx, yy, zz)
plt.show()


levels = np.linspace(0, 3, 20)


fig, ax = plt.subplots()
cs = ax.contour(xx, yy, zz, levels)
# ax.clabel(cs, inline=True, fontsize=10)


# Primeira derivada
def n_derive1(x1,x2):
    
    
    grad_x1 = (func(x1 + h, x2) - func(x1 - h, x2))/(2*h)
    grad_x2 = (func(x1, x2+h) - func(x1, x2-h))/(2*h)  
    grad = np.array([grad_x1, grad_x2])
    # grad = np.array([2*x1, 50*x2])
    return grad


def steepest(x, alpha, maxiter, tol):
    x1=x[0]
    x2=x[1]
    erro = 1e3
    i = 0
    devx=1e3
    # while i < maxiter:
    while  (devx>tol) or (erro>tol) :
        x_novo = x - alpha*n_derive1(x[0], x[1])       
        # print("x =", x)
        erro = abs (func( x_novo[0],x_novo[1] ) - func(x[0], x[1]) )
        devx = np.linalg.norm(x_novo-x)
        x = x_novo
        # plt.plot(x[0],x[1],'*', label='ótimo')
        xg1.append(x[0])
        xg2.append(x[1])
        if i>maxiter:
            break
        if x[0]>1e5 or x[1] >1e5:
            print('saindo por divergência')
            break
        # plt.show()
        i=i+1
        # i=i+1
        print("Iteração", i)
        print("erro=", erro)
        print("devx=", devx)
    return x_novo # no final retorna o último valor

# fig, ax = plt.subplots()
# Valores dos parâmetros
x0 = np.array([0.2,0.2])
alpha = 0.03
h = 1e-5
xg1=[]
xg2=[]

# Chamada da função:
    
# xopt-> ponto ótimo    
    
x_opt = steepest(x0, alpha, maxiter=500, tol=1e-6)


plt.plot(xg1,xg2,'r-')
plt.plot(xg1,xg2,'o')
plt.show()