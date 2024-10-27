import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt


# Definicação da Fobj
def func1(x1, x2):
    global teta
    y1= x1**2 + x2**2
    teta=1
    y1 = objective_function([x1,x2])
    return y1

# Definicação da Fobj
def func2(x1, x2):
    global teta
    teta=0
    y2 = objective_function([x1,x2])
    # y2 = (x1-4)**2 + (x2-4)**2
    return y2

# Definicação da Fobj
def objective_function(x):
    global teta
    y1= x[0]**2 + x[1]**2
    y2 = (x[0]-4)**2 + (x[1]-4)**2
    return teta*y1 + (1-teta)*y2


def func(x):
    global teta
    f = x[0]**2 + x[1]**2
    return f


lw = [-5 , - 5] 
up = [5, 5 ] 
global teta
teta = 1

D=2 # Dimensão do problema (número de variáveis de decisão)
k=0
P = 20 # numero de pontos para o Pareto
xp=np.zeros((D,P))
fp=np.zeros((D,P))
peso = np.linspace(0,1,P) # pesos para construção da curva de Pareto

# CHAMADA ITERATIVA DA OTIMIZAÇÃO PARA CADA PESO DE FUNÇÃO OBJETIVO
for i in peso:
    teta = i # peso de cada iteração (de 0 a 1)
    sol = dual_annealing(objective_function, bounds=list(zip(lw, up)))
    fp1 = func1(sol.x[0],sol.x[1]) # ARMAZENAMENTO DA FOBJ1
    fp2 = func2(sol.x[0],sol.x[1]) # ARMAZENAMENTO DA FOBJ2
    xp[:,k] = sol.x # ARMAZENAMENTO DAS SOLUÇÕES ÓTIMAS
    fp[:,k] = np.array([fp1,fp2])
    k+=1


# sol = dual_annealing(func, bounds=list(zip(lw, up)))

print(sol.fun)

# PLOTANDO AS CURVAS DE NÍVEL
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


levels = np.linspace(-25, 25, 50)


fig, ax = plt.subplots()
cs = ax.contour(xx, yy, zz1, levels)
cs = ax.contour(xx, yy, zz2, levels)
# ax.clabel(cs, inline=True, fontsize=10)

fig, axs = plt.subplots()
plt.plot(fp[0,:],fp[1,:],'*')
axs.set_title('Curva de Pareto')
axs.set(xlabel='Fobj1', ylabel='Fobj2')
# plt.show()

fig, axs = plt.subplots()
plt.plot(xp[0,:],xp[1,:],'o')
axs.set_title('Variáveis de Decisão')
axs.set(xlabel='x1', ylabel='x2')
# plt.show()

print('Valores otimos de x:', sol.x)
print('Valores otimos de Fobj(x):', sol.fun)
plt.show()       