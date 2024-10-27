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
    return x**2 - x

# Primeira derivada
def derive1(x):
    return 2*x - 1

# Segunda derivada
def derive2(x):
    return 2

# Função principal do Método de Newton
def newton(x0, maxiter=5000, tol=1e-4, func_tol=1e-4):
    x = x0  # estimativa inicial

    for i in range(maxiter):  # dentro do número de iterações...
        alpha = 1 / derive2(x)  # learning rate
        x_novo = x - alpha * derive1(x)  # atualização do valor
        
        # Verifica se a mudança é menor que a tolerância
        if abs(x_novo - x) < tol or abs(func(x_novo)-func(x)) < func_tol:
            print(f"Algoritmo convergiu para x = {x_novo} com f(x) = {func(x_novo)}")
            return x_novo  # retorna o valor atual
        
        x = x_novo  # atualiza valor para próxima iteração
        print(f"Iteração {i}: x = {x}, f(x) = {func(x)}")

    print("Número máximo de iterações alcançado.")
    return x_novo  # retorna o último valor

# Valores dos parâmetros
x0 = 3

# Chamada da função
x_opt = newton(x0)

# Plotagem
x = np.linspace(-10, 10, 400)
fig, ax = plt.subplots()
ax.plot(x, func(x), label="Função Objetivo")
ax.plot(x_opt, func(x_opt), marker='*', markersize=10, color='red', label="Ponto Ótimo")
ax.axhline(0, color='black', lw=0.5, ls='--')
ax.axvline(0, color='black', lw=0.5, ls='--')
ax.grid()
ax.set_xlabel("x")
ax.set_ylabel("Valor da Função Objetivo")
ax.legend()
plt.title("Método de Newton para Encontrar o Ponto Ótimo")
plt.show()
