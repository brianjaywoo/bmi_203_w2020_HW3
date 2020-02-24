import numpy as np
import pandas as pd
import sys
import pickle
from smith_waterman import algs
from smith_waterman import io

'''
Helper functions to test correctness of functions from smith_waterman/algs.py.
'''

"""
HW3

- Part 2
Implement/test and describe optimization algorithm (X/2)
Optimize starting from best matrix and explain (X/1)
Optimize starting from MATIO (X/1)
"""

"""
For smith-waterman:
1. Test that gap opening and extension is working correctly
2. Test that scoring matrix structure looks to be correct
3. Test that alignment is working; should look different for perfect sequence and scrambled sequences
(these are tested by running the entire module and the output is in my homework document)
"""

"""
For ROC curves (part 1):
1. Test that ROC lies between (0, 1)
2. Test that lowest point of the ROC curve is (0, 0) and the largest is (1, 1) (give or take)
3. Additional tests for functions in algs
"""

"""
For optimization (part 2):
1. Test structures of individuals in my population (for genetic algorithm optimization approach)
2. Test that symmetry is maintained
3. Test that new population looks different than first population
"""

#---------BEGIN PART 0: TESTING FUNCTIONS TO IMPLEMENT SMITH-WATERMAN---------#

def test_gap ():
    """
    Test gap penalty function.
    """
    #Test that gap penalty is of the form 'gap_start' + k * 'gap_extension', where
    #k is a length.
    assert algs.gap_penalty (0, 0, 0) == 0
    assert algs.gap_penalty (0, 8, 3) == 8
    assert algs.gap_penalty (10, 1, 1) == 11

def test_scoring_and_score_matrix ():
    """
    Test scoring function and score matrix output.
    Use multiple seq1, seq2 strings, as well as test score matrix structure.
    """
    #Perfect alignment
    seq1 = 'AAA' #5 for perfect match
    seq2 = 'AAA'
    BLOSUM_50 = io.BLOSUM_reader('BLOSUM50')
    test_matrix, test_pos, test_score  = algs.create_score_matrix(seq1, seq2, BLOSUM_50)
    assert isinstance(test_matrix, list)
    assert test_score == 15 #Same as 5 * 3 for perfect match
    assert test_pos == (3, 3) #Should be the the very bottom entry

    #Blank sequences
    seq1 = ''
    seq2 = ''
    test_matrix, test_pos, test_score  = algs.create_score_matrix(seq1, seq2, BLOSUM_50)
    assert isinstance(test_matrix, list)
    assert test_score == 0 #Empty sequences don't aign!
    assert test_pos == None

    #0 alignment
    seq1 = 'AAA'
    seq2 = 'RRR'
    test_matrix, test_pos, test_score  = algs.create_score_matrix(seq1, seq2, BLOSUM_50)
    assert isinstance(test_matrix, list)
    assert test_score == 0 #Bad alignment
    assert test_pos == None #No updating of the best position

    #Perfect alignment with gaps
    seq1 = 'AAAAAAAAAAAA' #12 long!
    seq2 = 'RRRRRRRRRRAAARRRRRRR'
    test_matrix, test_pos, test_score  = algs.create_score_matrix(seq1, seq2, BLOSUM_50)
    assert isinstance(test_matrix, list)
    assert test_score == 15 #Same as first situation!
    assert test_pos == (3, 13) #Lists first region in seq 1 where alignment begins perfectly

    #Flip the above situation to test symmetry
    seq2 = 'AAAAAAAAAAAA' #12 long!
    seq1 = 'RRRRRRRRRRAAARRRRRRR'
    test_matrix, test_pos, test_score  = algs.create_score_matrix(seq1, seq2, BLOSUM_50)
    assert isinstance(test_matrix, list)
    assert test_score == 15 #Same as first situation!
    assert test_pos == (13, 3) #Lists first region in seq 2 where alignment begins perfectly

    #Test the actual structure of the score matrix.
    test_df = pd.DataFrame (test_matrix)
    assert test_df.shape == (len(seq1) + 1, len(seq2) + 1) #Number of rows and cols should be len(seq1, seq2) + 1 respectively
    #Assert that values lie within this min and max
    assert test_df.values.min() >= 0
    assert test_df.values.max() <= test_score

#---------BEGIN PART 0: TESTING FUNCTIONS TO IMPLEMENT SMITH-WATERMAN---------#

#---------BEGIN PART 1: TESTING FUNCTIONS TO FIND OPT GAP PENALTY---------#

def test_roc():
    """
    Test that the roc data that is used to graph ROC's in part 1 and 2 is of the correct
    form and is being entered correctly.
    """
    #Load in a previously defined object
    with open("scoring_matrix_dictionary_for_roc.pkl", "rb") as g:
        test_roc = pickle.load(g)

    test_dict = {key: np.linspace(test_roc[key].values.max(), test_roc[key].values.min(), 200) for key in test_roc}
    test_roc_df = algs.return_roc_df (test_roc, test_dict)

    #Begin testing by looking at length of dictionary
    assert len(test_roc.keys()) == 5
    assert len(test_dict.keys()) == 5
    assert len(test_roc_df.keys()) == 5 #All of the objects that are used in logic to get to plotting ROC curve, are dictionaries

    #Test structure of one of the false-positive, true-positive rates dataframes present, for BLOSUM50
    BLOSUM_50_cut_df = test_roc_df['BLOSUM50']
    assert isinstance (BLOSUM_50_cut_df, pd.DataFrame)
    assert BLOSUM_50_cut_df.shape == (200, 2)

    #Test that values going into plotting ROC lie within [0, 1]
    assert BLOSUM_50_cut_df.values.min() >= 0
    assert BLOSUM_50_cut_df.values.max() <= 1


#---------END PART 1: TESTING FUNCTIONS TO FIND OPT GAP PENALTY---------#

#---------BEGIN PART 2: TESTING FUNCTIONS USED FOR OPTIMIZATION ---------#

def test_optimization ():
    """
    Test, using the BLOSUM50 matrix as input, that my optimization algorithm is
    proceeding as expected.
    """
    BLOSUM_50 = pd.DataFrame (io.BLOSUM_reader('BLOSUM50'))
    init_prob = 0.3
    pos_pairs = algs.parse_pairs('Pospairs.txt')
    neg_pairs = algs.parse_pairs('Negpairs.txt')

    #Begin testing: first see what happens if no mutation probability is given.
    #Then, all individuals should match the parent:

    populations = algs.initialize (BLOSUM_50, 0)

    assert len(populations) == 4 #The default I put in
    assert [i.equals(BLOSUM_50) for i in populations] == [True, True, True, True] #No mutation

    #Repeat but with a high initialization probability
    populations = algs.initialize (BLOSUM_50, init_prob)
    assert len(populations) == 4 #The default I put in
    assert [i.equals(BLOSUM_50) for i in populations] == [False, False, False, False] #All will be different

    #Test selection function
    test_list = [[1, 4], [2, 5], [3, 6], [4, 6]]
    selected, scores = algs.selection(test_list)

    assert len(selected) == 3 #5 is the low median
    assert selected == [2, 3, 4] #The selected individuals
    assert scores == [5, 6, 6] #The selected individuals' accompanying 'objective function' scores

    #Test repopulation method that structure of new generation looks about right
    #Do this with first two elem of populations object from above, to 'fix parents'

    new_gen = algs.repopulate (populations, 0) #fix the mutation rate; check structure
    assert isinstance (new_gen, list)
    assert len(new_gen) == 4

    #parent1_values = populations[0].values
    #parent2_values = populations[1].values
    #all_vals = np.append(parent1_values, parent2_values)




#---------END PART 2: TESTING FUNCTIONS USED FOR OPTIMIZATION ---------#
