# -*- coding: utf-8 -*-
"""
Universidade Federal Fluminense
Departamento de Engenharia Química e de Petróleo
Lizandro de Sousa Santos
E-mail: lizandrosousa@id.uff.br
"""

"""
Quasi-Newton Algorithm

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
    return (x+5)**2

# Primeira derivada
def derive1(x):
    return (func(x+h) - func(x-h)) / h

# Segunda derivada
def derive2(x):
    return (func(x+h) - 2*func(x) + func(x-h)) / (h**2)

def newton(xp, xq, maxiter, tol):
 
    yi=[]
    x_novo=[]
    x_old = []
    for i in range(maxiter): # dentro do número de iterações...
        
        alpha = 1
        
        m = ( derive1(xq) - derive1(xp) ) / (xq - xp)
        x_novo = xq - alpha*(derive1(xq)/m) # atualização do valor
        A = derive1(x_novo)
        B = derive1(xq)
        C = derive1(xp)
        
        if (A*B)<0:
            xp=x_novo
        else:
            xq=x_novo
            
        if abs(x_novo-xq) < tol: # se a tolerância for estabelecida...
            print("algoritmo convergiu para x=", x_novo)
            yi.append(x_novo)
            
            fig, ax = plt.subplots()
            xi = np.linspace(-10, 10)
            yii = np.array(yi)
            ax.plot(xi,func(xi))
            ax.plot(yii,func(yii),marker='o')
            ax.grid()
            
            return x_novo # retorna o valor atual
            break # interrompe
        # x_old = x # valor antigo de x    
        # x = x_novo # caso contrário, atualiza valor e continua
        print("Iteração", i)
        print("x =", x)
        yi.append(x) # gravando os valores..
    fig, ax = plt.subplots()
    xi = np.linspace(-10, 10)
    yii = np.array(yi)
    ax.plot(xi,func(xi))
    ax.plot(yii,func(yii),marker='o')
    return x # no final retorna o último valor

# Valores dos parâmetros
xq = 3
xp = -3
h=1e-6
# Chamada da função:
    
# xopt-> ponto ótimo    
    
x_opt = newton(xp, xq, maxiter=5000, tol=0.0001)

#Plotagem
x = np.linspace(-10, 10)
fig, ax = plt.subplots()
ax.plot(x,func(x))
ax.plot(x_opt, func(x_opt), marker='o', label="ponto ótimo")
ax.grid()
ax.set_xlabel("x")
ax.set_ylabel("objective function")
ax.legend()