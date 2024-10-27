from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around



 # Definição da Função Objetivo
def obj(x):
   return x[0]**2.0 + x[1]**2.0

# Fator de Mutação
def mutation(x, F):
   return x[0] + F * (x[1]- x[2])

# Definição dos Bounds
def check_bounds(mutated, bounds):
   mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
   return mutated_bound

# Operação de Cruzamento
def crossover(mutated, target, dims, cr):
 # Valor randômico uniforme
 p = rand(dims)
 # vetor de cruzamento randômico
 trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
 return trial

  # initialise population of candidate solutions randomly within the specified bounds
def differential_evolution(pop_size, bounds, iter, F, cr):
   pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1]- bounds[:, 0]))
   # evaluate initial population of candidate solutions
   obj_all = [obj(ind) for ind in pop]
   # find the best performing vector of initial population
   best_vector = pop[argmin(obj_all)]
   best_obj = min(obj_all)
   prev_obj = best_obj
   # run iterations of the algorithm
   for i in range(iter):
     # iterate over all candidate solutions
    for j in range(pop_size):
        # choose three candidates, a, b and c, that are not the current one
        candidates = [candidate for candidate in range(pop_size) if candidate != j]
        a, b, c = pop[choice(candidates, 3, replace=False)]
        # perform mutation
        mutated = mutation([a, b, c], F)
        # check that lower and upper bounds are retained after mutation
        mutated = check_bounds(mutated, bounds)
        # perform crossover
        trial = crossover(mutated, pop[j], len(bounds), cr)
        # compute objective function value for target vector
        obj_target = obj(pop[j])
        # compute objective function value for trial vector
        obj_trial = obj(trial)
        # perform selection
        if obj_trial < obj_target:
            # replace the target vector with the trial vector
            pop[j] = trial
            # store the new objective function value
            obj_all[j] = obj_trial
            # find the best performing vector at each iteration
    best_obj = min(obj_all)
            # store the lowest objective function value
    if best_obj < prev_obj:
        best_vector = pop[argmin(obj_all)]
        prev_obj = best_obj
        # report progress at each iteration
        print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
   return [best_vector, best_obj]
 # define population size
pop_size = 10
# define lower and upper bounds for every dimension
bounds = asarray([(-5.0, 5.0), (-5.0, 5.0)])
# define number of iterations
iter = 100
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7

# perform differential evolution
solution = differential_evolution(pop_size, bounds, iter, F, cr)
print('\nSolution: f([%s]) = %.5f' % (around(solution[0], decimals=5), solution[1]))