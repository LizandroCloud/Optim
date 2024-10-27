# -*- coding: utf-8 -*-
"""
Universidade Federal Fluminense
Departamento de Engenharia Química e de Petróleo
Lizandro de Sousa Santos
E-mail: lizandrosousa@id.uff.br
"""

"""
Basic Newton Algorithm

x0-> initial estimative
derive1 -> first derivative 
derive2 -> second derivative
alpha   -> learning rate (stepsize)
maxiter -> maximum iteration number
tol     -> tolerance  

"""

import numpy as np
import matplotlib.pyplot as plt
from examples import *

def fplot(func, opt, xi, x_opt, xpmin, xpmax):
    if opt=='max':
        s=-1
    else:
        s=1

    x = np.linspace(xpmin, xpmax, 400)
    fig, ax = plt.subplots()
    ax.plot(x, s*func(x), label="Objective function")
    ax.plot(xi, s*func(xi), marker='o', label="Decision variables trajectory")
    ax.plot(x_opt, s*func(x_opt), marker='*', markersize=10, color='red', label="Optimal point")
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.axvline(0, color='black', lw=0.5, ls='--')
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("Objective function")
    ax.legend()
    plt.title("Objective function x Decision variable")
    plt.show()
    return []


# First derivative
def derive1(func,x,h):
    dfdx = ( ( func(x+h) - func(x-h) ) / (2*h) )  # Central Finitte Difference
    return dfdx

# Second Derivative
def derive2(func,x,h):
    df2dx2 = ( func(x+h) - 2*func(x) + func(x-h) ) / (h**2) # Central Finitte Difference
    return df2dx2

# Newton Method
def newton(func, x0, maxiter, tol, func_tol, h, opt, xpmin, xpmax):
    if opt=='max': # flag to verify if the problem is maximization or minimization
        s=-1
    else:
        s=1
    xi = np.linspace(xpmin, xpmax) # range of x vector used to plot Fobj vs decision variable
    x = x0  # initial estimative

    for i in range(maxiter):  # iterations according to the maxiter
             
        alpha = 0.1 / derive2(func,x,h)  # learning rate (stepsize)
        x_novo = x - alpha * derive1(func, x,h)  # actualing the x value
        
        # Verify the convergence criteria
        if abs(x_novo - x) < tol or abs(func(x_novo)-func(x)) < func_tol: 
            print(f"The algorithm converged to x = {x_novo} with f(x) = {s*func(x_novo)}")           
            #yii = np.array(yi) # Storing the 
            fplot(func, opt, yii, x_novo, xpmin, xpmax) # function used to plot Fobj vs Decision variable

            return x_novo  # returning the last optimal value
        
        x = x_novo  # actualizing the decision variable for the next iteration
        print(f"Iteração {i}: x = {x}, f(x) = {s*func(x)}")
        print("Iteração", i)
        print("x =", x)
        yi.append(x) # storing the optimal values..
        yii = np.array(yi)

    
    print("Número máximo de iterações alcançado.")
    return x_novo  # return the optimal value

# Valores dos parâmetros
yi=[]
x0 = 1.1
opt = 'min'
maxiter=5000
tol=1e-4
func_tol=1e-4
h=1e-5
xpmin = -10
xpmax  = 12

x_opt = newton(uofpA, x0, maxiter, tol, func_tol, h, opt, xpmin, xpmax)

