# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 06:08:38 2024

@author: lizst
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:26:05 2024

@author: lizst
"""

"""
Universidade Federal Fluminense
Departamento de Engenharia Química e de Petróleo
Lizandro de Sousa Santos
E-mail: lizandrosousa@id.uff.br
"""

"""
Algoritmo Enxame de Partículas Aplicado a Multi-Objetivos

"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Definicação da Fobj
def func1(x1, x2):
    global teta
    teta=1
    y1 = objective_function([x1,x2])
    return y1

# Definicação da Fobj
def func2(x1, x2):
    global teta
    teta=0
    y2 = objective_function([x1,x2])
    return  y2

# Definicação da Fobj
def objective_function(x):
    global teta

    
    y1 = x[0]**2 + x[1]**2
    y2 = (x[0]-1)**2 + x[1]**2
    return teta*y1 + (1-teta)*y2


def PSO(N,D,c1,c2,tmax,xmin,xmax, teta):
    #Inicialização dos parâmetros
    x=np.zeros((N,D))
    X=np.zeros(N)
    p=np.zeros((N,D)) # melhor posicao
    P=np.zeros(N) # melhor valor de fobj
    v=np.zeros((N,D))
    for i in range(N): # iteracao para cada particula (Iteracao Inicial)
        for d in range(D):
            x[i,d]=xmin[d]+(xmax[d]- xmin[d])*np.random.uniform(0,1) # inicializaão da posicao
            v[i,d]=0 # inicializaão da velocidade (dx)
        X[i]= objective_function(x[i,:])
        p[i,:]=x[i,:] 
        P[i]=X[i]
        
        if i==0:
            g=np.copy(p[0,:])   ############
            G=P[0] # valor global da fobj
        
        if P[i]<G:
            g=np.copy(p[i,:])    ####################
            G=P[i] # gravando valor de fobj da particula i
    
    # Plotando o domínio de cálculo
    fig, axs = plt.subplots(2, 2, gridspec_kw={'hspace': 0.7, 'wspace': 0.7})
    axs[0, 0].plot(x[:,0],x[:,1],'ro')
    axs[0, 0].set_title('Iteracao Inicial')
    
    #Iterações
    tmax=500
    for tatual in range(tmax):
        for i in range(N):
            R1=np.random.uniform(0,1) # valor randomico para R1
            R2=np.random.uniform(0,1) # valor randomico para R2
            #  cálculo do fator de inercia /(varia com a iteracao)
            wmax=0.9 
            wmin=0.4
            w=wmax-(wmax-wmin)*tatual/tmax  # fator de inercia 
            
            v[i,:]=w*v[i,:]+ c1*R1*(p[i,:]-x[i,:])+c2*R2*(g-x[i,:]) # velocidade 
            x[i,:]=x[i,:]+v[i,:] # posicao
            
            for d in range(D): # garantia de limites
                if  x[i,d]<xmin[d]:
                    x[i,d]=xmin[d]
                    v[i,d]=0
                
                if x[i,d]>xmax[d]:
                    x[i,d]=xmax[d]
                    v[i,d]=0
            
            X[i]=objective_function(x[i,:])
            if X[i]<P[i]:
                p[i,:]=x[i,:]  # melhor posicao da particula i
                P[i]=X[i]  # atualização de P (melor fobj de i)
                if P[i]< G: # verificaão se é melhor que a fobj global
                    g=np.copy(p[i,:])  # gravando melhor posicao global
                    G=P[i]
                    
        if tatual==49:
            axs[0, 1].plot(x[:,0],x[:,1],'ro')
            axs[0, 1].set_title('Iteracao 20')
                
        if tatual==99:
            axs[1, 0].plot(x[:,0],x[:,1],'ro')
            axs[1, 0].set_title('Iteracao 100')
            
        if tatual==499:
            axs[1, 1].plot(x[:,0],x[:,1],'ro')
            axs[1, 1].set_title('Iteracao 499')
                
              
    for ax in axs.flat:
        ax.set(xlabel='x1', ylabel='x2')
    return g, objective_function(g)

xmin=np.array([-5,-5]) # Limites Mínimos
xmax=np.array([5,5]) # Limites Máximos
# H=abs(xmax-xmin) # Diferença entre máximo e mínimo
N=50 # Número de Partículas
# Parâmetros de Sintonia
c1=0.8 # individual
c2=1.2 # coletivo
D=2 # Dimensão do problema (número de variáveis de decisão)
tmax = 500 # Número Máximo de Iterações
k=0
P = 10 # numero de pontos para o Pareto
xp=np.zeros((D,P))
fp=np.zeros((D,P))
peso = np.linspace(0,1,P) # pesos para construção da curva de Pareto
global teta


for i in peso:
    teta = i # peso de cada iteração (de 0 a 1)
    x, f = PSO(N,D,c1,c2,tmax,xmin,xmax, teta) # chamada do PSO da iteração atual
    fp1 = func1(x[0],x[1])
    fp2 = func2(x[0],x[1])
    xp[:,k] = x
    fp[:,k] = np.array([fp1,fp2])
    k+=1
    

fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')

xa = np.linspace(-5,5,100)
xb = np.linspace(-5,5,100)
xx, yy = np.meshgrid(xa,xb)
zz1 = func1(xx,yy)
zz2 = func2(xx,yy)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Fobj')
ax.plot_surface(xx, yy, zz1)
ax.view_init(elev=40, azim=45)
# plt.show


levels = np.linspace(-5, 5, 50)


fig, ax = plt.subplots()
cs = ax.contour(xx, yy, zz1, levels)
cs = ax.contour(xx, yy, zz2, levels)
ax.set_title('Curvas de Nível')
ax.set(xlabel='x', ylabel='y')
# ax.clabel(cs, inline=True, fontsize=10)

fig, axs = plt.subplots()
plt.plot(fp[0,:],fp[1,:], '*')
axs.set_title('Curva de Pareto')
axs.set(xlabel='Fobj1', ylabel='Fobj2')
# plt.show()

fig, axs = plt.subplots()
plt.plot(xp[0,:],xp[1,:], 'o')
axs.set_title('Variáveis de Decisão')
axs.set(xlabel='x1', ylabel='x2')
# plt.show()

print('Valores otimos de x:', x)
print('Valores otimos de Fobj(x):', f)
plt.show()       