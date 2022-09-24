---
title: Genetic Algorithm and Multi-Object Optimization
tags: ["GA", "MOO", "Optimization", "MathModel", "Python", "Matlab"]
date: 2022-09-03 09:37:01
---
## Reference

- [Matlab: Constrained MOO using GA ver.2](https://www.mathworks.com/matlabcentral/fileexchange/29806-constrained-moo-using-ga-ver-2)
- [Wiki: MOO](https://en.wikipedia.org/wiki/Multi-objective_optimization)
- [Zhihu: GA](https://www.zhihu.com/question/23293449)

## GA

__Genetic Algorithm(GA)__ is a widely used way to solve __NP optimization problems__. It simulates the natural process to get a approximately optimized result. Advantages of it contains simple, fast and so on, but it has the disadvantages many 'simulation algorithms' have, like local optimization, collapsing(or __premature__, typically) and so on.

- __premature__: the gene which performs better diffuses too fast to make other genes survive to try.

## MOO

__Multi-Objective Optimization(MOO)__ is a common kind of problem in real world, which has multiple object functions(probably non-linear) and constrains.

It has a basic 'template' math representation of:
$$
Objects: \min_{x \in X}(f_1(x), f_2(x), ..., f_n(x))
$$
The set $X$ above means the __feasible__ set of the decision vector $x$. $x \in X$ is called a __feasible decison__.  

## By Matlab

The codes below came from the reference above. And I will try to make a pythonic one. Commets in these matlab codes are  explicit enough to understand.

```matlab
%% Simple EMOO problem
% The objective is to find the pareto front of the MOO problem defined as follows:
% Maximize:
% f1(X) = 2*x1 + 3*x2
% f2(X) = 2/x1 + 1/x2
%   such that:
%     10 > x1 > 20
%     20 > x2 > 30
%
% Author: Wesam Elshamy
%
% PhD candidate, Kansas State University
%
% welshamy@ksu.edu
%
% http://cis.ksu.edu/~welshamy
%%
clear;
clc;
% Define parameters
iterations = 500;
population_size = 500;
mutation_rate = 0.02;
crossover_rate = 0.3;
population = zeros(population_size,3);
% Initialize population within constraints
for i = 1 : population_size
    x1 = (rand*10 + 10); % x1 value for individual i within range
    x2 = (rand*10 + 20); % x2 value for individual i within range
    population(i,1) = x1;
    population(i,2) = x2;
    population(i,3) = 2*x1 + 3*x2; % f1 value for individual i
    population(i,4) = 2/x1 + 1/x2; % f2 value for individual i
end
% Iterations
for iter = 1 : iterations
    pool = population;
    for i = 1 : population_size
        % -------------- crossover ----------------
        if (rand < crossover_rate)
            parent1 = pool(randi(size(pool,1)),:); % randomly select parent1
            parent2 = pool(randi(size(pool,1)),:); % randomly select parent2
            child1 = [parent1(1) parent2(2) zeros(1,2)];
            child2 = [parent1(2) parent2(1) zeros(1,2)];
            pool = [pool; child1; child2];
        end
        
        % -------------- mutation ------------------
        if (rand < mutation_rate)
            individual = pool(randi(size(pool,1)),:); % randomly select individual from pool
            bit = randi(2); % select gene to mutate
            if (bit == 1)
                individual(1) = rand*10 + 10; % value of mutation respects x1 constraints
            else
                individual(2) = rand*10 + 20; % value of mutation respects x2 constraints
            end
            individual(3:4) = zeros(1,2); % assign temporary fitness of zeros
            pool = [pool; individual]; % add individual to pool
        end
    end
    
    % fitness evaluation of the new individuals
    for i = population_size+1 : size(pool,1)
        pool(i,3) = 2*x1 + 3*x2;
        pool(i,4) = 2/x1 + 1/x2;
    end
    
    temp_pop = [];
    % select non dominated individuals to start next iteration with
    for i = 1 : size(pool,1)
        dominated = false;
        for j = 1 : size(pool,1)
            if (pool(i,3)<pool(j,3) && pool(i,4)<pool(j,4)) % if individual i is dominated by individual j
                dominated = true;
                break; % break and go to next individual
            end
        end
        if (~dominated) % if individual not dominated
            temp_pop = [temp_pop; pool(i,:)]; % add it to the pool
            if (size(temp_pop,1) == population_size) % Have enough individuals to fill populatino array?
                break;
            end
        end
    end
    population = temp_pop;
end
% visualization of the results
disp('x1 and x2 values for non-dominated solutions:')
disp(population(:,[1,2]))
f = population(:,[3,4]); % store f1 and f2 values for the population in f
plot(f(:,1), f(:,2), 'x'); % plot the Pareto front
title({'Pareto front of: Max:', 'f_1(X) = 2x_1 + 3x_2', 'f_2(X) = 2/x_1 + 1/x_2'});
xlabel('f_1(X)');
ylabel('f_2(X)');

```

## By Python

It is meant to learn to do MOO by GA using Python resembling the matlab codes above.

```python
#!/bin/python3
#-*-coding:UTF-8-*-
#Author: LeafLight
#Date: 2022-09-03 22:34:18
#---
# Simple MOO
# the objective is to find the pareto front of the MOO defined as follows:
# Maxmize:
# f1(X) = 2 * x1 + 3 * x2
# f2(X) = 2 / x1 + 1 / x2
# 	such that:
# 		10 < x1 < 20
#		20 < x2 < 30
# Reference: [MOO by GA using matlab](https://www.mathworks.com/matlabcentral/fileexchange/29806-constrained-moo-using-ga-ver-2) by Wesam Elshamy

from tqdm import tqdm
import numpy as np
from numpy.random import rand, randint, uniform
# number to iter(or evolution)
iterations = 500
# number of the population
population_size = 500
# probability to mutate
mutation_ = 0.02
# probability to cross over
crossover = 0.3
# parameters(genes) of popupation
# population_size * [x1, x2, f1, f2]
def obj_func(X):
	f1 = 2 * X[0] + 3 * X[1]
	f2 = 2 / X[0] + 1 / X[1]
	return np.array([f1, f2])
def st(X):
	const1 = X[0] > 10
	const2 = X[0] < 20
	const3 = X[1] > 20
	const4 = X[1] < 30
	return all([const1, const2, const3, const4])
low_bounds = [-40, -40]
up_bounds  = [40, 40]
class GA:
    def __init__(self, iterations=500, population_size=500, crossover=0.3, mutation=0.02, st_width=1e3, split_point=1,
               var_num=2, obj_num=2, obj_func=obj_func, st=st, lb=low_bounds, ub=up_bounds, no_bounds=False):
        # hyperparameters
        self.iterations = iterations
        self.population_size = population_size
        self.crossover = crossover
        self.mutation = mutation
        self.st_width = st_width
        self.split_point=1
        # objective and constraints
        self.var_num = var_num
        self.obj_num = obj_num
        self.obj_func = obj_func
        self.lb = lb
        self.ub = ub
        self.st = st
        self.no_bounds = no_bounds

    def evolution(self, ):
		########################################
		# util func
        def get_val_val():
            # get valid value
            while True:
                if self.no_bounds:
                    X = rand(self.var_num) * 2 * self.st_width - self.st_width
                else:
                    X = uniform(self.lb, self.ub)

                if self.st(X):
                    break
            return X

		########################################
        # evolution main
        population = np.zeros([self.population_size, self.var_num + self.obj_num])
        # Init the population within constraints rand
        init_pop_bar = tqdm(range(population_size))
        init_pop_bar.set_description("Init Population")
        for i in init_pop_bar:
            population[i, :self.var_num] = get_val_val()
            population[i, self.var_num:] = self.obj_func(population[i, :self.var_num])
        print("Population Init complete !")

        iter_p_bar = tqdm(range(self.iterations))
        iter_p_bar.set_description("Evolution######")
        for it in iter_p_bar:
            pool = population

			# General Steps every iter.
			## 0. init the pool
			## 1. cross over
			## 2. mutation
			## 3. choose non-dominated individuals
            for i in range(self.population_size):
            # -------------------- Cross Over --------------------
                if rand(1) < self.crossover:
					# Choose 2 parents randomly
                    parent1 = pool[randint(self.population_size)]
                    parent2 = pool[randint(self.population_size)]
                    # Make children
                    child1_var = np.concatenate([parent1[:self.split_point], parent2[self.split_point:self.var_num]])
                    child2_var = np.concatenate([parent2[:self.split_point], parent1[self.split_point:self.var_num]])
                    child1_obj = self.obj_func(child1_var)
                    child2_obj = self.obj_func(child2_var)
                    child1 = np.concatenate([child1_var, child1_obj])
                    child2 = np.concatenate([child2_var, child2_obj])
                    # append children to the pool
                    pool = np.vstack([pool, np.vstack([child1, child2])])
			# -------------------- Mutations --------------------
                if rand(1) < self.mutation:
                    # randomly choose the one to mutate
                    individual_index = randint(self.population_size)
                    # randomly mutate gene
                    mutated_X = get_val_val()
                    # randomly mutate var index
                    mutated_bit = randint(self.var_num)
                    # mutate
                    pool[individual_index, mutated_bit] = mutated_X[mutated_bit]
                    # evaluate the fitness of the mutated individual
                    pool[individual_index, self.var_num:] = self.obj_func(pool[individual_index, :self.var_num])
			# -------------------- Choose Non-dominated Individuals --------------------
			# init the temp pool
            temp_pool = np.zeros([self.population_size, self.var_num + self.obj_num])
			# loop over all the individuals
            non_dominated_cnt = 0
            for p_focus in pool:
                dominated = False
				# to see if there is other individuals dominate it
                for p_to_compare in pool:
                    if all(np.greater(p_to_compare[self.var_num:], p_focus[self.var_num:])):
                        dominated = True	
                        break
				# if it's non-dominated, add it to the temp pool
                if ~dominated:
                    temp_pool[non_dominated_cnt] = p_focus
                    non_dominated_cnt += 1
                    if non_dominated_cnt == self.population_size:
                        break
                    population = temp_pool
        print("###GA complete!###")
        return population

if __name__ == '__main__':
    ga = GA()
    test_res = ga.evolution()
    print(test_res)
