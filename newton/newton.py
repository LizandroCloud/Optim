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
derive1 -> primeira derivada da função
derive2 -> segunda derivada da função
alpha -> passo da otimização (learning rate)
maxiter -> número máximo de iterações
tol -> tolerância para convergência    

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Função objetivo
def func(x):
    return x**2 - x

# Primeira derivada
def derive1(x):
    return 2*x - 1

# Segunda derivada
def derive2(x):
    return 2

def newton(x0, maxiter, tol):
    x=x0 # estimativa inicial

    for i in range(maxiter): # dentro do número de iterações...
        
        alpha = 1/derive2(x)
            
        x_novo = x-alpha*derive1(x) # atualização do valor
        if abs(x_novo-x) < tol: # se a tolerância for estabelecida...
            print("algoritmo convergiu para x=", x_novo)
            return x_novo # retorna o valor atual
            break # interrompe
        x = x_novo # caso contrário, atualiza valor e continua
        print("Iteração", i)
        print("x =", x)
    return x_novo # no final retorna o último valor

yi=[]
x_novo=[]

# Valores dos parâmetros
x0 = 3

# Chamada da função:
    
# xopt-> ponto ótimo    
    
x_opt = newton(x0, maxiter=5000, tol=0.0001)

#Plotagem
x = np.linspace(-10, 10)
fig, ax = plt.subplots()
ax.plot(x,func(x))
ax.plot(x_opt, func(x_opt), marker='*', label="ponto ótimo")
ax.grid()
ax.set_xlabel("x")
ax.set_ylabel("objective function")
ax.legend()