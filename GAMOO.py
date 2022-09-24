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
def default_obj_func(X):
	f1 = 2 * X[0] + 3 * X[1]
	f2 = 2 / X[0] + 1 / X[1]
	return np.array([f1, f2])
def default_st(X):
	const1 = X[0] > 10
	const2 = X[0] < 20
	const3 = X[1] > 20
	const4 = X[1] < 30
	return all([const1, const2, const3, const4])
low_bounds = [-40, -40]
up_bounds  = [40, 40]
class GA:
    def __init__(self, iterations=500, population_size=500, crossover=0.3, mutation=0.02, st_width=1e3, split_point=1,
               var_num=2, obj_num=2, obj_func=default_obj_func, st=default_st, lb=low_bounds, ub=up_bounds, no_bounds=False):
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
    #ga_default = GA()
    #default_res = ga_default.evolution()
    #print(default_res)
    # -------------------- Test ---------------------
    # edit the parameters below to custom a GAMOO
    # all you have to specify
    # - [] test_obj_func
    # - [] test_st
    # - [] test_lb
    # - [] test_ub
    # - [] test_ub
    # - [] var_num
    # - [] obj_num
    def test_obj_func(X):
        f1 = 3 * X[0] + 4 * X[1] - 10 * X[2]
        f2 = 1 / X[0] + 10 / X[1] - 2 / X[2]
        f3 = 2 / X[0] + 5 / X[1] - 2 / X[2]
        return f1, f2, f3
    def test_st(X):
        const1 = (20 * X[0] + X[1]) < 10
        const2 = (10 * X[1] + X[2]) < 30
        return all([const1, const2])

    test_lb = (-40, -40, -40)
    test_ub = (40, 40, 40)
    ga_test = GA(iterations=500, population_size=500, crossover=0.3, mutation=0.02, st_width=1e3, split_point=1,
               var_num=3, obj_num=3, obj_func=test_obj_func, st=test_st, lb=test_lb, ub=test_ub, no_bounds=False)
    test_res = ga_test.evolution()
    print(test_res)
