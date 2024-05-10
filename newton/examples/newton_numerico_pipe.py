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
def func(d):
    return 500 + 240*d**1.5 + 600*d**(-2.5) + 27*(d**-4)

# Primeira derivada
def derive1(x,h):
    
    dfdx = ( ( func(x+h) - func(x-h) ) / (2*h) ) 
    return dfdx

# Segunda derivada
def derive2(x,h):
    
    df2dx2 = ( func(x+h) - 2*func(x) + func(x-h) ) / (h**2)
    return df2dx2

def newton(x0, maxiter, tol, h):
    x=x0 # estimativa inicial

    for i in range(maxiter): # dentro do número de iterações...
        
        alpha = 1/derive2(x,h)
            
        x_novo = x-alpha*derive1(x,h) # atualização do valor
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
x0 = 2

# Chamada da função:
    
# xopt-> ponto ótimo    
    
x_opt = newton(x0, maxiter=5000, tol=0.0001, h=1e-5)

#Plotagem
x = np.linspace(1, 3)
fig, ax = plt.subplots()
ax.plot(x,func(x))
ax.plot(x_opt, func(x_opt), marker='*', label="ponto ótimo")
ax.grid()
ax.set_xlabel("x")
ax.set_ylabel("objective function")
ax.legend()