#Here Specify python Version

import numpy as np
import matplotlib.pyplot as plt
from random import randint
from random import uniform
from timeit import default_timer as timer
#import pandas as pd
#import time
#from tqdm import tqdm
import os
import sys
#import multiprocessing
#from multiprocessing import Process
#from time import sleep
#from multiprocessing import Pool, cpu_count
#from timeit import default_timer as timer
#import scipy
#from scipy.signal import find_peaks, peak_prominences
#from numba import jit
from Evo.Cell import Cell
#from memory_profiler import profile
from Evo.Basic_activations import *
epistatic_matrix=np.array([[0, 1, 0.1, 0.1],[1,0,0.1,0.1], [0.1, 0.1, 0, 1],[0.1,0.1,1,0]])
def create_epistatic_matrix(num_genes, corr_factor_1=0.1, corr_factor_2=1):
    epistatic_matrix=np.full((num_genes, num_genes),corr_factor_1)
    np.fill_diagonal(epistatic_matrix, 0)
    i=1
    while i<epistatic_matrix.shape[0]:
        j=i-1
        loc_1=[i, j]
        loc_2=[j, i]
        epistatic_matrix[loc_1[0], loc_1[1]]=corr_factor_2
        epistatic_matrix[loc_2[0], loc_2[1]]=corr_factor_2
        i+=2
        j=i-1
    return epistatic_matrix
def evaluate_original(ind, epistatic_matrix=epistatic_matrix):
    sig_ind=tanh(ind)
    #sig_ind=sig_ind.reshape(1,4)
    sig_ind=sig_ind.T
    temp=0
    for i in range(0, sig_ind.shape[1]):
        value=np.matmul(sig_ind[:,i], sig_ind)
        #print(value)
      
        value=np.sum(np.multiply(value, epistatic_matrix[i,:]))
        #print((value))
        temp=temp+(value)
    fitness=temp/2
    return fitness
def create_condition(num_genes):
    condition=np.random.uniform(-1,1, [num_genes,])
    return condition
def calculate_max_size_list(list_of_cells):
    max_size_list=[]
    for i in range(len(list_of_cells)):
        max_size_list.append(list_of_cells[i][-1])
    return max_size_list
def calculate_size(ind_exspression, current_weight):
    return sigmoid((np.matmul(ind_exspression.T, current_weight))/100)
def grow_2(cells, time, condition, initial_area=1.0, display_results=True,  Hebbian=False):
     timespan=np.linspace(0.1,time,100)
     individuals=[[] for e in range(len(cells))]
     individuals_exp=[[] for e in range(len(cells))]
     for t in timespan:
         area=initial_area
         for cell in cells:
             if Hebbian==True:
                for i in range(100):
                    cell.Regulation_update()
             cell.Calculate_exspression()
             
         growth_factors=[calculate_size(getattr(x, "EXSPRESSION_MATRIX"),current_weight=condition)[0] for x in cells]
         for i in range(len(individuals)):
            area_inc=growth_factors[i]*area*t
            area+=area_inc
            individuals[i].append(area)
     return individuals
def make_first_gen(Prog_Cell, ev_distance, gen_size):
    original_matrix=Prog_Cell.EXSPRESSION_MATRIX
    num_genes=Prog_Cell.genes
    first_gen=[Cell(num_genes, EXSPRESSION_MATRIX=np.copy(original_matrix)) for d in range(gen_size)]
    for i in range(ev_distance):
        for item in first_gen:
            item.Basic_mutation()
    return first_gen
#@profile
def evolution_main(first_gen, num_generations, gen_size, condition, seed, mode="Regulatory",save="few",save_freq=100, log=False, Log_index=1):
    np.random.seed(seed)
    new_pop=[first_gen[x].Make_offspring(gen_size=gen_size) for x in range(len(first_gen))]
    max_size_list=[]
    #generation_size=[]
    population_trajectories=[[] for e in range(len(new_pop))]
    population_exspression=[[] for e in range(len(new_pop))]
    population_regulation=[[] for e in range(len(new_pop))]
    start=timer()
    time_2=0
    for generation in range(num_generations):   
        if generation%100==0 and log==True:
            time=timer()
            traj_size=(sys.getsizeof(population_trajectories[0])*len(new_pop))/1000
            f=open("Log_file_{}.txt".format(Log_index), "a")
            f.write("Generation {} Time from start {} Run time {}\n Trajectories size {} One traj {} \n".format(generation,np.round(time-start, decimals=2),np.round(time-time_2, decimals=2),sys.getsizeof(population_trajectories)/1000, traj_size))
            f.close()
            time_2=time
        for i in range(len(new_pop)):
            for j in range(len(new_pop[i])):
                new_pop[i][j].Basic_mutation()
                if mode=="Regulatory":
                    new_pop[i][j].Regulation_mutation()
        next_gen=[]
        for i in range(len(new_pop)):
            grown=grow_2(new_pop[i], 5, condition, display_results=False)
            grown_size=calculate_max_size_list(grown)
            #generation_size.append([grown_size])
            pop_max=max(grown_size)
            if save=="all":
                population_trajectories[i].append(pop_max)
                population_exspression[i].append(new_pop[i][np.argmax(grown_size)].EXSPRESSION_MATRIX)
                population_regulation[i].append(new_pop[i][np.argmax(grown_size)].REGULATION_MATRIX)
                #next_gen.append(new_pop[i][np.argmax(grown_size)])
            if save=="few":
                if generation==num_generations-1 or generation%save_freq==0:
                    population_exspression[i].append(new_pop[i][np.argmax(grown_size)].EXSPRESSION_MATRIX)
                    population_regulation[i].append(new_pop[i][np.argmax(grown_size)].REGULATION_MATRIX)
                    #next_gen.append(new_pop[i][np.argmax(grown_size)])
                population_trajectories[i].append(pop_max)
            if save=="last":
                if generation==num_generations-1:
                    population_exspression[i].append(new_pop[i][np.argmax(grown_size)].EXSPRESSION_MATRIX)
                    population_regulation[i].append(new_pop[i][np.argmax(grown_size)].REGULATION_MATRIX)
                    #next_gen.append(new_pop[i][np.argmax(grown_size)]) 
                population_trajectories[i].append(pop_max)
            next_gen.append(new_pop[i][np.argmax(grown_size)])
        next_1=[next_gen[x].Make_offspring(gen_size=gen_size) for x in range(len(first_gen))]
        for i in range(len(new_pop)):
            for j in range(len(new_pop[i])):
                new_pop[i][j].End_life()
        new_pop=next_1

        #new_pop=[next_gen[x].Make_offspring(gen_size=gen_size) for x in range(len(first_gen))]
        #print("Next Gen ", np.array(next_gen).shape, "New_pop  ", np.array(new_pop).shape)
    return population_trajectories, population_regulation, population_exspression
def generate_progenitor(num_genes, condition, display_results=False, grow_time=5):
    cells=[Cell(num_genes) for d in range(20)]
    grown_cells=grow_2(cells, grow_time, condition, display_results=False)
    max_size_list=[cell[-1] for cell in grown_cells]
    index=np.random.choice(max_size_list, 1)
    while index<0:
        index=np.random.choice(max_size_list,1)[0]
        #print("looking")
    else:
        pass
    cell_ind=max_size_list.index(index)
    Prog_Cell=cells[cell_ind]
    if display_results==True:
        plt.figure(figsize=(10,15))
        for j in individuals:
            plt.plot(timespan, j, alpha=0.3)
            plt.plot(timespan, individuals[cell_ind], alpha=1, linewidth=2, c="r")
        plt.title("Selected Individual", fontweight="bold") 
    return Prog_Cell, index
#stack data split by the paralelisation back together
def unpacker(results):
    traj=[]
    reg=[]
    exp=[]
    population_trajectories=[results[i][0] for i in range(len(results))]
    population_trajectories=np.array(population_trajectories)
    for i in range(population_trajectories.shape[0]):
        for j in range(population_trajectories.shape[1]):
            traj.append(population_trajectories[i][j])
    regulation=[results[i][1] for i in range(len(results))]
    regulation=np.array(regulation)
    for i in range(regulation.shape[0]):
        for j in range(regulation.shape[1]):
            reg.append(regulation[i][j][0])
    exspression=[results[i][2] for i in range(len(results))]
    exspression=np.array(exspression)
    for i in range(exspression.shape[0]):
        for j in range(exspression.shape[1]):
            exp.append(exspression[i][j][0])
    return traj, reg, exp
def best_traj(trajectories):
    fin_point=[traj[-1] for traj in trajectories]
    norm=min(fin_point)
    normalized=[num/norm for num in fin_point]
    return normalized
def split_gen(gen, num_treads):
    split=np.array_split(gen, num_treads)
    split=[list(split[i]) for i in range(len(split))]
    return split
#Unpacker for situation when multiple regulation/exspression values are saved
def unpacker_multiple(results):
    trajectories=[]
    population_trajectories=[results[i][0] for i in range(len(results))]
    population_trajectories=np.array(population_trajectories)
    for i in range(population_trajectories.shape[0]):
        for j in range(population_trajectories.shape[1]):
            trajectories.append(population_trajectories[i][j])
    regulation=np.array([np.squeeze(results[i][1]) for i in range(len(results))])
    exspression=np.array([np.squeeze(results[i][2]) for i in range(len(results))])
    return np.array(trajectories), regulation, exspression

