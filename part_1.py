#!/usr/bin/env python3

'''
Part 1 of the BMI 203 W2020 HW3 assignment. For a high-level overview: I will
find the optimal gap opening and extension penalty for a given true positive rate,
and then will use this optimal gap penalty to find the optimal scoring matrix
by looking at ROC curves.
'''
#Modules contain code that imports and reads in data from other files. Scripts are
#made to be directly executed.

from smith_waterman.algs import create_score_matrix, traceback, next_move, alignment_string, print_matrix, parse_pairs, score_generator, explore_gap_parameters, tp_or_fp_search, return_score_df, return_roc_df, gap_plot, roc
from smith_waterman.io import single_line_fasta_reader, BLOSUM_reader
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import sys

# Make sure that program is called correctly
if len(sys.argv) < 5:
    #-E, -R stand for evaluation, or run
    print("Usage: python part_1.py [-E| -R] <path_to_pos_pairs> <path_to_neg_pairs> <score_matrix>")
    sys.exit('Exiting because of incorrect arguments')

print('Reading in path now...')

#Get system arguments
logic = sys.argv[1]
pos_path = sys.argv[2]
neg_path = sys.argv[3]
input_matrix = BLOSUM_reader (sys.argv[4])

#Parse the positive pairs and negative pairs file
pos_pairs = parse_pairs(pos_path)
neg_pairs = parse_pairs(neg_path)


#We now want to only look at true positive rates (TP) of 0.7. At this TP rate, what is
#the corresponding false positive rate?

#To do this, I will simply return the data point in my true positive column corresponding
#to TP = 0.7 for each gap penalty combination and set it as a threshold. Using this
#threshold, I then return the corresponding false positive rate (defined as the number of
#neg pairs that is above the corresponding TP=0.7 threshold).

#Evaluation logic: evaluate objects only once and then save them for future loading.
if logic == '-E':
    #Make a dataframe with the maximum score lists of positive and negative pairs.
    #I only consider gap starts from 1-20 with steps of 2: saves time for slow
    #non-parallel processing
    gap_starts = range(1, 20, 2)
    gap_extensions = range(1, 5)

    #Calls to evaluate: long!
    print('Evaluation start for gap exploration.. ')
    dict_gaps = explore_gap_parameters(pos_pairs, neg_pairs, gap_starts, gap_extensions, input_matrix)

    #Make a new dictionary with same keys as the dictionary with scores, and populate just with false positive scores
    key_values = dict_gaps.keys()
    dict_fp = {keys: [] for keys in key_values}

    for key in dict_gaps:
        dict_fp[key] = tp_or_fp_search(dict_gaps, key)

    #Save the resulting dictionaries of scores for each gap penalty combination,
    #and false positive rates
    with open("bmi_203_w2020_dictionary_of_gap_with_scores.pkl", "wb") as f:
        pickle.dump (dict_gaps, f)

    with open("bmi_203_w2020_dictionary_of_fp.pkl", "wb") as f:
        pickle.dump (dict_fp, f)

#Otherwise: I assume I've run evaluation, and I load in previous data

elif logic == '-R':
    print('Loading in previously defined objects.. ')

    with open("bmi_203_w2020_dictionary_of_gap_with_scores.pkl", "rb") as g:
        dict_gaps = pickle.load(g)

    with open("bmi_203_w2020_dictionary_of_fp.pkl", "rb") as g:
        dict_fp = pickle.load(g)

    #print('Loaded pos dictionary looks like', dict_fp) #test code only

    #Code to graph out the false positive rate as a function of the gap penalties

    #Finding the minimum key (corresponding to the best gap penalty)
    best_fp = min(dict_fp.values())
    print('best false positive rate is', best_fp)

    #Parse the dictionary of false positive rates for the corresponding best gap combination
    best_gap = [i for i in dict_fp.keys() if dict_fp[i] == best_fp]

    #It's a tuple so get the start and extension
    gap_start = best_gap[0][0]
    gap_extension = best_gap[0][1]
    print ('Corresponding gap penalty is', (gap_start, gap_extension))

    #Plotting out the best false positive rate by gap penalties

    #Get keys as x-axis
    keys = list(dict_fp.keys())
    keys = [str(i) for i in keys]

    #Get values as y-axis
    values = list(dict_fp.values())

    #Plot the resulting fp rates by keys
    gap_plot (keys, values)

#Let's now feed in multiple scoring matrices using this (11, 1) gap penalty.

#Similar logic as above: will only evaluate the actual roc dictionary once for each
#scoring matrix; then will save it.

if logic == '-E':
    dict_roc = {}
    #All of the five scoring matrices fed in
    for i in ('BLOSUM50', 'BLOSUM62', 'MATIO', 'PAM100', 'PAM250'):
        score_matrix = BLOSUM_reader (i)
        combined_df = return_score_df (neg_pairs, pos_pairs, score_matrix, gap_start = gap_start, gap_extension = gap_extension )
        #Add the dataframe of scores to the dictionary with keys being scoring matrices
        dict_roc[i] = combined_df

    #print('false positive dictionary looks like', dict_roc)

    #Save the results
    with open("scoring_matrix_dictionary_for_roc.pkl", "wb") as f:
        pickle.dump (dict_roc, f)

#Else just load the previous dict_roc object

elif logic == '-R':
    print('Loading in previously defined objects')
    with open("scoring_matrix_dictionary_for_roc.pkl", "rb") as g:
        dict_roc = pickle.load(g)

    #Call to roc function to handle the logic by each dictionary handle. Then graph all lines together on one graph.
    range_dict = {key: np.linspace(dict_roc[key].values.max(), dict_roc[key].values.min(), 200) for key in dict_roc}
    dict_for_roc = return_roc_df (dict_roc, range_dict)
    print('The dictionary of dataframes looks like', dict_for_roc)

    #Plot each roc curve (5 total) on the same graph
    roc(dict_for_roc)

#So it looks like PAM100 is the best scoring matrix. Let's see what happens if we
#normalize.

#Let's now normalize the scores for just two situations: let's choose the best scoring matrix we had previously and
#generate scores both with and without normalization. Then, plot the ROC curves for these two matrices.

#Calculate for PAM100
optimal = 'PAM100'
matrix = BLOSUM_reader (optimal)

#Calculate non-normalized and normalized dfs, with evaluation only once

if logic == '-E':
    #We've actually calculated this before, but re-do for consistency's sake
    nonnorm_df = return_score_df (neg_pairs, pos_pairs, matrix, gap_start, gap_extension)
    norm_df = return_score_df (neg_pairs, pos_pairs, matrix, gap_start, gap_extension, normalize = True)

    #Make dictionary
    norm_dict = {'Non-normalized': nonnorm_df, 'Normalized': norm_df}

    with open("dict_PAM100_scores_norm_vs_nonnorm.pkl", "wb") as f:
        pickle.dump (norm_dict, f)

elif logic == '-R':
    with open("dict_PAM100_scores_norm_vs_nonnorm.pkl", "rb") as g:
        norm_dict = pickle.load(g)

    #Similar logic as above with plotting ROC for 5 different scoring matrices.
    norm_range_dict = {key: np.linspace(norm_dict[key].values.max(), norm_dict[key].values.min(), 200) for key in norm_dict}
    dict_for_norm_roc = return_roc_df(norm_dict, norm_range_dict)
    #Plot and save the norm vs unnorm figure
    roc(dict_for_norm_roc, fig_name = 'BMI_203_W2020_ROC_comparison_norm_vs_unnorm_PAM100.png', title = 'Unnormalized vs normalized ROC, PAM100')
    sys.exit('Part 1 complete!')
