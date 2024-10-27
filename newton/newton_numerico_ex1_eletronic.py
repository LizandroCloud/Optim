# -*- coding: utf-8 -*-
"""
Universidade Federal Fluminense
Departamento de Engenharia Química e de Petróleo
Lizandro de Sousa Santos
E-mail: lizandrosousa@id.uff.br
"""

"""
Basic Newton Algorithm

x0-> estimativa inicial
derive1 -> primeira derivada da função
derive2 -> segunda derivada da função
alpha -> passo da otimização (learning rate)
maxiter -> número máximo de iterações
tol -> tolerância para convergência    

"""

import numpy as np
import matplotlib.pyplot as plt

# Função objetivo
def func(x):
    return -( 125 - 50*x  + 5*x**2 ) * 100 / ( (1 + 0.5*(1.5*x**-3)*(0.25) )**4 )

# Primeira derivada
def derive1(x,h):
    dfdx = ( ( func(x+h) - func(x-h) ) / (2*h) ) 
    return dfdx

# Segunda derivada
def derive2(x,h):
    df2dx2 = ( func(x+h) - 2*func(x) + func(x-h) ) / (h**2)
    return df2dx2

# Função principal do Método de Newton
def newton(x0, maxiter=5000, tol=1e-4, func_tol=1e-5, h=1e-6):
    x = x0  # estimativa inicial

    for i in range(maxiter):  # dentro do número de iterações...
        alpha = 1 / derive2(x,h)  # learning rate
        x_novo = x - alpha * derive1(x,h)  # atualização do valor
        
        # Verifica se a mudança é menor que a tolerância
        if abs(x_novo - x) < tol or abs(func(x_novo)-func(x)) < func_tol:
            print(f"Algoritmo convergiu para x = {x_novo} com f(x) = {func(x_novo)}")
            return x_novo  # retorna o valor atual
        
        x = x_novo  # atualiza valor para próxima iteração
        print(f"Iteração {i}: x = {x}, f(x) = {func(x)}")

    print("Número máximo de iterações alcançado.")
    return x_novo  # retorna o último valor

# Valores dos parâmetros
x0 = 1.5

# Chamada da função
x_opt = newton(x0)

# Plotagem
x = np.linspace(0.5, 6, 400)
fig, ax = plt.subplots()
ax.plot(x, -func(x), label="Função Objetivo")
ax.plot(x_opt, -func(x_opt), marker='*', markersize=10, color='red', label="Ponto Ótimo")
ax.axhline(0, color='black', lw=0.5, ls='--')
ax.axvline(0, color='black', lw=0.5, ls='--')
ax.grid()
ax.set_xlabel("x")
ax.set_ylabel("Valor da Função Objetivo")
ax.legend()
plt.title("Método de Newton para Encontrar o Ponto Ótimo")
plt.show()
