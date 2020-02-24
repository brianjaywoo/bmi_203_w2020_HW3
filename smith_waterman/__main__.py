#!/usr/bin/env python3

#Heavily adapted from https://gist.github.com/radaniba/11019717; most of the
#original code lies in visualizing the actual alignment string


'''A Python implementation of the Smith-Waterman algorithm for local alignment
of amino acid strings. Takes in two sequence paths, and an associated scoring matrix,
to return an alignment string.
'''

from .algs import create_score_matrix, traceback, next_move, alignment_string, print_matrix
from .io import single_line_fasta_reader, BLOSUM_reader
import os
import sys

# Make sure the program is called correctly
if len(sys.argv) < 4:
    print("Usage: python -m smith_waterman <seq_1> <seq_2> <input_matrix>")
    sys.exit('Incorrect usage')

#Open up the files that I will use for alignment.

print('Reading in files now...')

seq1 = single_line_fasta_reader (sys.argv[1])
seq2 = single_line_fasta_reader (sys.argv[2])
input_matrix = BLOSUM_reader (sys.argv[3])

#Read input system arguments and return in proper format
print('input seq1 looks like', seq1)
print('input seq2 looks like', seq2)
print('the input amino acid reisdue match matrix looks like', input_matrix[0], input_matrix[1])

# Initialize the scoring matrix.
score_matrix, max_pos, max_score = create_score_matrix(seq1, seq2, input_matrix)

#Print the complete score matrix
print_matrix (score_matrix)
print ('The position to be followed in traceback is', max_pos)
print ('The maximum score is', max_score)

# Traceback. Find the optimal path through the scoring matrix. This path
# corresponds to the optimal local sequence alignment.
seq1_aligned, seq2_aligned = traceback(seq1, seq2, score_matrix, max_pos)
assert len(seq1_aligned) == len(seq2_aligned), 'aligned strings should be the same size'

# Pretty print the results. The printing follows the format of BLAST results
# as closely as possible.

#This logic was largely left untouched from original repo
alignment_str, idents, gaps, mismatches = alignment_string(seq1_aligned, seq2_aligned)
alength = len(seq1_aligned)
print(' Identities = {0}/{1} ({2:.1%}), Gaps = {3}/{4} ({5:.1%})'.format(idents,
      alength, idents / alength, gaps, alength, gaps / alength))

#BLAST-like handling of printing the results
for i in range(0, alength, 60):
    seq1_slice = seq1_aligned[i:i+60]
    print('Query  {0:<4}  {1}  {2:<4}'.format(i + 1, seq1_slice, i + len(seq1_slice)))
    print('             {0}'.format(alignment_str[i:i+60]))
    seq2_slice = seq2_aligned[i:i+60]
    print('Sbjct  {0:<4}  {1}  {2:<4}'.format(i + 1, seq2_slice, i + len(seq2_slice)))
    print()
