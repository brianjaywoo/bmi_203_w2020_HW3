#!/usr/bin/env python3

'''
Part 2 of the BMI 203 W2020 HW3 assignment. High-level overview: to find
the optimal scoring matrix for this list of positive and negative pairs (leads
to greatest separation between pos/neg pair calls), I will
implement a genetic algorithm.
'''

from smith_waterman.algs import parse_pairs, initialize, assess_fitness, selection, mutation, repopulate, tp_or_fp_search, return_score_df, return_roc_df, roc
from smith_waterman.io import single_line_fasta_reader, BLOSUM_reader
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import sys

#Part 1: Identifying and initializing solutions

#An initial population would look like a population of 'individual scoring matrices',
#that are randomly generated. The objective function is given by the following:

#-- max (sum(TP) for FP in [0, 0.1, 0.2, 0.3]) --#

#Make sure that all arguments present in script

if len(sys.argv) < 5:
    #Similar logic as part 1: evaluate objects only with -E flag, otherwise load them.
    #The scoring matrix included is used as the unoptimized matrix that will be
    #iteratively improved during optimization.
    print("Usage: python part_2.py [-E| -R] <path_to_pos_pairs> <path_to_neg_pairs> <score_matrix>")
    sys.exit('Exiting because of incorrect arguments')

logic = sys.argv[1]
pos_path = sys.argv[2]
neg_path = sys.argv[3]
unopt_matrix = pd.DataFrame (BLOSUM_reader (sys.argv[4]))

#I will use the gap and extension penalties derived previously, so a gap combination
#of (11, 1).

gap_start = 11
gap_extension = 1

#Parse the positive pairs and negative pairs file
pos_pairs = parse_pairs(pos_path)
neg_pairs = parse_pairs(neg_path)

#Initialize a bunch of parameters: 'init_prob' is the probability with which
#an entry in the unoptimized matrix will be mutated randomly, as an initial individual.

#number_term is the number of iterations of the genetic algorithm.

#fp_range is used for the objective function and is the false positive threshold considered.

init_prob = 0.3
number_term = 2
fp_range = [0, 0.1, 0.2, 0.3]

print('Beginning optimization...')

#For ease of saving names; part 2 only asks to consider optimal and MATIO matrix, hence
#the limited considerations here for naming
if sys.argv[4] == 'PAM100':
    name = 'PAM100_optimal_matrix.pkl'
elif sys.argv[4] == 'MATIO':
    name = 'MATIO_optimal_matrix.pkl'

#Actually only optimize if this flag is passed in
if logic == '-E':

    #Initialize a population of (4) matrices from a given starting matrix
    populations = initialize (unopt_matrix, init_prob)

    #Selection of 'fit' individuals using objective function values

    fitness = []

    #Start selection, genetic operations; terminate at iteration # = number_term
    for i in range(number_term):
        print ('Optimization cycle is', i, 'and progress is', i/number_term * 100, '% complete')
        #Calculate objective function score for each scoring matrix initialized individual.
        for score_matrix in populations:
            #The 'fitness' list has, for each element, a list with entry 1 = matrix, 2 = its associated OBJECTIVE FUNCTION score.
            fitness.append ([score_matrix, assess_fitness(neg_pairs, pos_pairs, score_matrix, gap_start, gap_extension, fp_range)])

        #From selection, get individuals that are above the median objective function score as threshold.
        populations, scores = selection (fitness)

        #Termination condition
        if i == number_term - 1:
            best_score = max(scores)
            #Select optimal matrix based on whether the objective function value is the highest.
            optimal_matrix = [i[0] for i in fitness if i[1] == best_score]

            #Logic to handle if there are 2+ matrices with exact same objective function value
            if len(optimal_matrix) > 1:
                optimal_matrix = optimal_matrix[0] #Arbitrarily take the first elem if all of them are equally good
                assert len([optimal_matrix]) == 1, 'There can only be one optimal matrix.'

            #Printing help
            print ('The optimal matrix optimized from', name, 'looks like', optimal_matrix, 'and its associated score was', best_score)

            #Save the final matrix with name defined previously
            with open(name, "wb") as f:
                pickle.dump (optimal_matrix, f)

            sys.exit('Optimal matrix calculation complete!')

        #If no termination: iterate again. Repopulate the selected individuals using
        #a combination of recombination (of two individuals) and mutation.
        
        #Lose the scores at this point; get a list of score matrices again
        populations = repopulate (populations)
        fitness = []

elif logic == '-R':
    with open(name, 'rb') as g:
        optimal_matrix = pickle.load (g)

    #Plot ROC curves for original and new optimized matrices that I select

    orig_df  = return_score_df (neg_pairs, pos_pairs, unopt_matrix, gap_start, gap_extension)
    optimized_df = return_score_df (neg_pairs, pos_pairs, optimal_matrix, gap_start, gap_extension)
    opt_dict = {'Unoptimized': orig_df, 'Optimized': optimized_df}

    opt_range_dict = {key: np.linspace(opt_dict[key].values.max(), opt_dict[key].values.min(), 200) for key in opt_dict}
    dict_for_opt_roc = return_roc_df(opt_dict, opt_range_dict)

    fig_name = sys.argv[4] + ' optimal_vs_unoptimal.png'
    roc(dict_for_opt_roc, fig_name = fig_name, title = 'Optimized vs unoptimized scoring matrix ROC' +  sys.argv[4])

    sys.exit ('Optimization and ROC graphs complete!')
