# -*- coding: utf-8 -*-
"""
Universidade Federal Fluminense
Departamento de Engenharia Química e de Petróleo
Lizandro de Sousa Santos
E-mail: lizandrosousa@id.uff.br
"""

"""
Quadratic-Interpolation Algorithm

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
    k1 = 5/6
    k2 = 5/3
    k3 = 1/6
        
    cas1 = -(k1+x)/(2*k3)
    cas2 = (((k1+x)**2 + 4*k3*x)**0.5)/(2*k3)
    cas = cas1 + cas2
    cbs = k1*cas/(x+k2)
    f1 = cbs
    return -f1


def quadratic(x1,x3,tol,maxiter):
 
    yi=[]
    xZ=1e99
    k=0
    # Geração dos Pontos Iniciais
    
    x2 = 0.5*(x1 + x3)
    f1 = func(x1)
    f2 = func(x2)
    f3 = func(x3)
    z1 = (x2 - x3)*f1
    z2 = (x3 - x1)*f2
    z3 = (x1 - x2)*f3
    z4 = (x2 + x3)*z1+(x3 + x1)*z2+(x1 + x2)*z3
    x_novo = z4/(2*(z1 + z2 + z3))
    f_novo = func(x_novo)
    
    for i in range(maxiter): # dentro do número de iterações...      
          
        while abs(xZ-x_novo) >= tol: # se a tolerância não for estabelecida...
            if (x1 < x_novo) and (x_novo < x2):
                if f_novo <= f2:
                    x3 = x2
                    f3 = f2
                    x2 = x_novo
                    f2 = f_novo
                else:
                    x1=x_novo
                    f1=f_novo
            elif (x2 < x_novo) and (x_novo < x3):
                if f_novo <= f2:
                    x1 = x2
                    f1 = f2
                    x2 = x_novo
                    f2 = f_novo
                else: 
                    x3 = x_novo
                    f3 = f_novo
            yi.append(x_novo)
            yii = np.array(yi)
  
            xZ = x_novo        
            z1 = (x2 - x3)*f1
            z2 = (x3 - x1)*f2
            z3 = (x1 - x2)*f3
            z4 = (x2 + x3)*z1 + (x3 + x1)*z2 + (x1 + x2)*z3
            x_novo = z4/(2*(z1 + z2 + z3))
            f_novo = func(x_novo)
            d = abs(xZ - x_novo)
            print("Iteração", k)
            print("x =", x_novo)

            k=k+1
        fig, ax = plt.subplots()
        xi = np.linspace(0, 5)
        yii = np.array(yi)
        ax.plot(xi,-func(xi))
        ax.plot(yii,-func(yii),marker='o')
        ax.grid() 
        return x_novo # retorna o valor atual
            

  

# Valores dos parâmetros
x1 = 0.1
x3 = 4

# Chamada da função:
    
# xopt-> ponto ótimo    
    
x_opt = quadratic(x1, x3, maxiter=50000, tol=1e-4)

