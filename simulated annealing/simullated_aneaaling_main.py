# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:43:35 2022

@author: lizst
"""

"""
Universidade Federal Fluminense
Departamento de Engenharia Química e de Petróleo
Lizandro de Sousa Santos
E-mail: lizandrosousa@id.uff.br
"""

"""
Recozimento Simulado (Simulated Annealing)


"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

pmin = -1.5
pmax = 1.5

# Definindo a função-objetivo
def f(x):
    x1 = x[0]
    x2 = x[1]
    obj =  100*(x[1]-(x[0])**2)**2 +(x[0]-1)**2
    return obj

# Estimativa Inicial
x_start = [1.2, 1.2]

# Configurando a Estimativa inicial (Estado inicial)
i1 = np.arange(-2.0, 3.0, 0.01)
i2 = np.arange(-2.0, 3.0, 0.01)
x1m, x2m = np.meshgrid(i1, i2)
fm = np.zeros(x1m.shape)
for i in range(x1m.shape[0]):
    for j in range(x1m.shape[1]):
        fm[i][j] = 0.2 + x1m[i][j]**2 + x2m[i][j]**2 \
             - 0.1*math.cos(6.0*3.1415*x1m[i][j]) \
             - 0.1*math.cos(6.0*3.1415*x2m[i][j])

# Curvas de Nível
plt.figure()
CS = plt.contour(x1m, x2m, fm)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Non-Convex Function')
plt.xlabel('x1')
plt.ylabel('x2')

##################################################
# Início do Algoritmo
##################################################
# Número de Ciclos (Iterações)
n = 100
# Número de Tentativas por Ciclos
m = 10
# Número de Soluções Aceitas
na = 0.0
# Probabilidade de Aceitar Uma Solução Ruim no Início
p1 = 0.2
# PProbabilidade de Aceitar Uma Solução Ruim no Final
p50 = 0.001
# Temperatura Inicial
t1 = -1.0/math.log(p1)
# Temperatura Final
t50 = -1.0/math.log(p50)
# Redução da Temperatura em Cada Ciclo
frac = (t50/t1)**(1.0/(n-1.0))
# Inicialização de x
x = np.zeros((n+1,2))
x[0] = x_start
xi = np.zeros(2)
xi = x_start
na = na + 1.0
# Gravando os melhores resultados
xc = np.zeros(2)
xc = x[0] # Variável de Decisão
fc = f(xi)
fs = np.zeros(n+1)
fs[0] = fc # Função-Objetivo
# Temperatura Atual
t = t1
# Média da Energia
DeltaE_avg = 0.0
for i in range(n):
    print('Ciclo: ' + str(i) + ' com Temperatura: ' + str(t))
    for j in range(m):
        # Novos Pontos 
        xi[0] = xc[0] + random.random() - 0.5
        xi[1] = xc[1] + random.random() - 0.5
        # Avaliação das Restrições
        xi[0] = max(min(xi[0],pmax),pmin)
        xi[1] = max(min(xi[1],pmax),pmin)
        DeltaE = abs(f(xi)-fc)
        if (f(xi)>fc):
            # Inicializar DeltaE_avg se uma solução pior foi encontrada
            #   na primeira iteração
            if (i==0 and j==0): DeltaE_avg = DeltaE
            # Função-objetivo é pior
            # Probabilidade de aceitar:
            p = math.exp(-DeltaE/(DeltaE_avg * t))
            # Determinar se aceitou ou não
            if (random.random()<p):
                # Aceitar a pior solução
                accept = True
            else:
                # Não aceitar
                accept = False
        else:
            # Aceitar automaticamente se a função é menor
            accept = True
        if (accept==True):
            # Atualização das soluções
            xc[0] = xi[0]
            xc[1] = xi[1]
            fc = f(xc)
            # Incremento das soluções
            na = na + 1.0
            # Atualizar DeltaE_avg
            DeltaE_avg = (DeltaE_avg * (na-1.0) +  DeltaE) / na
    # Gravar os melhores valores de x
    x[i+1][0] = xc[0]
    x[i+1][1] = xc[1]
    fs[i+1] = fc
    # Atualização da T para o ciclo seguinte
    t = frac * t

# Imprimir solução
print('Melhor solução: ' + str(xc))
print('Melhor função-objetivo: ' + str(fc))

plt.plot(x[:,0],x[:,1],'y-o')
plt.savefig('contornos.png')

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(fs,'r.-')
ax1.legend(['FUnção-objetivo'])
ax2 = fig.add_subplot(212)
ax2.plot(x[:,0],'b.-')
ax2.plot(x[:,1],'g--')
ax2.legend(['x1','x2'])

# Salvando a figura como PNG
plt.savefig('iteracoes.png')

plt.show()
