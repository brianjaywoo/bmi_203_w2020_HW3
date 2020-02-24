from .io import single_line_fasta_reader, BLOSUM_reader
import matplotlib.pyplot as plt
import statistics
import random
import numpy as np
import pandas as pd

#---------BEGIN PART 0: DEFINING FUNCTIONS TO IMPLEMENT SMITH-WATERMAN---------#

#Function that calculates affine gap penalty for any length k.

def gap_penalty (k, gap_start, gap_extension):
    #These parameters are fed in
    return gap_start + (k * gap_extension)

#Function that calculates score for any cell (i, j) in my scoring matrix, by considering
#0, the similarity score, all possible gap penalty top, all possible gap penalty left.
def calc_score(seq1, seq2, match_matrix, score_matrix, x, y, gap_start, gap_extension):
    '''Calculate score for a given x, y position in the scoring matrix.
    The score is based on the up, left, and upper-left neighbors, or 0.
    '''
    aa_seq1 = seq1[x - 1]
    aa_seq2 = seq2[y - 1]

    #Some of the fasta files aren't capitalized with 'x', so I do that manually here
    if aa_seq1 == 'x':
        aa_seq1 = 'X'

    if aa_seq2 == 'x':
        aa_seq2 = 'X'

    if isinstance(match_matrix, list):
        match_matrix = match_matrix[0] #??? Don't know why sometimes my match matrix comes in list format
    similarity = match_matrix.loc[aa_seq1, aa_seq2]

    #Compute the diagonal score
    diag_score = score_matrix[x - 1][y - 1] + similarity

    #Initialize lists for left and top search
    horizontal, vertical = [], []

    #Return the max of the gap penalty in the horizontal direction
    for k in range(0, y):
        #Search all score matrix possibilities
        left_score = score_matrix[x][k] - gap_penalty(y - 1 - k, gap_start, gap_extension)
        horizontal.append(left_score)
    left_score = max(horizontal)

    #Return the max of the gap penalty in the vertical direction
    for k in range(0, x):
        up_score = score_matrix[k][y] - gap_penalty(x - 1 - k, gap_start, gap_extension)
        vertical.append(up_score)
    up_score = max(vertical)

    #Return the max of all the options
    return max(0, diag_score, up_score, left_score)

#Create a score matrix that takes in two sequences, a scoring matrix, and will return
#3 things: the score matrix, the maximal score position of the score matrix (for traceback),
#and the maximal score.

def create_score_matrix(seq1, seq2, match_matrix, gap_start = 5, gap_extension = 2, normalize = False):
    '''Create a matrix of scores representing trial alignments of the two sequences.
    Sequence alignment can be treated as a DP problem. This function
    creates a graph (2D matrix) of score. The path with the highest cummulative score is the
    best alignment.
    '''
    #Initialize a scoring matrix for use in the algorithm. Need to include an
    #additional row and column for purposes of being able to calculate each position's score.
    rows, cols = len(seq1) + 1, len(seq2) + 1
    score_matrix = [[0 for col in range(cols)] for row in range(rows)]

    # Fill the scoring matrix.
    max_score = 0
    max_pos   = None    # The row and column of the highest score in matrix; to be updated
    #'i' is the iterator thru rows, 'j' is the iterator thru cols
    for i in range(1, rows):
        for j in range(1, cols):
            #Call the previous calc score on position (i, j)
            score = calc_score(seq1, seq2, match_matrix, score_matrix, i, j, gap_start, gap_extension)
            #Update the maximal score and position
            if score > max_score:
                max_score = score
                max_pos   = (i, j)
            #Fill in the scoring matrix
            score_matrix[i][j] = score

    #Only update max score if normalize flag is passed
    if normalize == True:
        max_score = max_score / min ([rows, cols]) #Divide by length of the shorter sequence (+ 1)
    return score_matrix, max_pos, max_score

#This function performs traceback through the matrix starting from the maximal position
#found previously. Movement then goes either up, left, or up-left, corresponding to
#a match/mismatch (diag), up or left (gap).

def traceback(seq1, seq2, score_matrix, max_pos):
    '''Find the optimal path through the matrix.
    This function traces a path from the max-pos to the top-left corner of
    the scoring matrix. Each move corresponds to a match, mismatch, or gap in one
    or both of the sequences being aligned. Moves are determined by the score of
    three adjacent squares: the upper square, the left square, and the diagonal
    upper-left square.
    WHAT EACH MOVE REPRESENTS
        diagonal: match/mismatch
        up:       gap in sequence 1
        left:     gap in sequence 2
    '''
    #These options are flags for 'where to go next' for the algorithm
    END, DIAG, UP, LEFT = range(4)
    aligned_seq1 = []
    aligned_seq2 = []

    #Start at the maximal score position in the scoring matrix
    x, y         = max_pos

    #Call the 'next move' helper function to reposition x and y
    move         = next_move(score_matrix, x, y)

    #If not at the top left, do the following:
    while move != END:
        if move == DIAG:
            #Move up and left = diagonally
            aligned_seq1.append(seq1[x - 1])
            aligned_seq2.append(seq2[y - 1])

            #Decrement x and y to denote that we're now in the diagonal square
            x -= 1
            y -= 1

        elif move == UP:
            aligned_seq1.append(seq1[x - 1])

            #'-' denotes gaps
            aligned_seq2.append('-')

            #Decrement x only, keep y the same
            x -= 1
        else:

            #Reverse the logic as above
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[y - 1])
            y -= 1

        #Iterate through the matrix until we hit the end
        move = next_move(score_matrix, x, y)

    #After the while loop, append the last two amino acid residues
    aligned_seq1.append(seq1[x - 1])
    aligned_seq2.append(seq1[y - 1])

    #The aligned sequences will be a mix of amino acid residues and gaps (in the case
    #of the two sequences not being perfectly identical)

    return ''.join(reversed(aligned_seq1)), ''.join(reversed(aligned_seq2))

#An 'up' move corresponds to decrementing 'x' and keeping 'y' constant.
#A 'left' move corresponds to decrementing 'y' and keeping 'x' constant.

def next_move(score_matrix, x, y):
    #Set up what diagonal, up and left moves look like in terms of matrix notation.
    diag = score_matrix[x - 1][y - 1]
    up   = score_matrix[x - 1][y]
    left = score_matrix[x][y - 1]

    #Logic to handle movement: let's talk about what this is doing. 'diag', 'up',
    #and 'left', are the three options to follow. If the diag score is greater
    #than both the gap scores, then move up. If 0, then the sequence is done, and
    #return END.
    if diag >= up and diag >= left:     # Note TIE favors daig move
        return 1 if diag != 0 else 0    # 1 signals a DIAG move. 0 signals the end.
    elif up > diag and up >= left:      # Tie goes to UP move.
        return 2 if up != 0 else 0      # UP move or end.
    elif left > diag and left > up:
        return 3 if left != 0 else 0    # LEFT move or end.
    else:
        # Execution should not reach here.
        raise ValueError('invalid move during traceback')


def alignment_string(aligned_seq1, aligned_seq2):
    '''Construct a special string showing identities, gaps, and mismatches.
    This string is printed between the two aligned sequences and shows the
    identities (|), gaps (-), and mismatches (:). As the string is constructed,
    it also counts number of identities, gaps, and mismatches and returns the
    counts along with the alignment string.
    AAGGATGCCTCAAATCGATCT-TTTTCTTGG-
    ::||::::::||:|::::::: |:  :||:|   <-- alignment string
    CTGGTACTTGCAGAGAAGGGGGTA--ATTTGG
    '''
    # Build the string as a list of characters to avoid costly string
    # concatenation.

    idents, gaps, mismatches = 0, 0, 0
    alignment_string = []
    #Consider cases in the aligned sequence:
    for aa1, aa2 in zip(aligned_seq1, aligned_seq2):
        #Perfect match
        if aa1 == aa2:
            alignment_string.append('|')
            idents += 1
        #This handles a gap.
        elif '-' in (aa1, aa2):
            alignment_string.append(' ')
            gaps += 1
        #This is the mismatch case.
        else:
            alignment_string.append(':')
            mismatches += 1
    #Return the joined alignment string.
    return ''.join(alignment_string), idents, gaps, mismatches


def print_matrix(matrix):
    '''Print the scoring matrix.
    Make a pandas dataframe and print it.
    '''
    printed_matrix = pd.DataFrame(matrix)
    print('The score matrix looks like', printed_matrix)

#---------END PART 0: DEFINING FUNCTIONS TO IMPLEMENT SMITH-WATERMAN---------#

#---------BEGIN PART 1: FUNCTIONS TO FIND OPTIMAL GAP COMBO/ SCORE MATRIX ---------#

#Helper function: given a path to the neg or positive pairs file, returns a pairs list
#with each entry in the list being a tuple of the pair sequences. Call this on
#both the positive and negative pairs files.

def parse_pairs(path):

    with open(path) as f:
        pairs = f.readlines()
        #Strip off the newline characters
        pairs = [tuple(i.strip().split()) for i in pairs]

    return pairs

#Given a list of tuples of pairs, and an input scoring matrix, returns a list of
#max scores for each pair. Specifically, performs alignment between the two sequences.

def score_generator (pair_list, input_matrix, gap_start = 5, gap_extension = 2, normalize = False):

    #Iterate through each tuple in the list of lists
    list_of_scores = []

    i = 0 #Test code only; comment out if would like to fully iterate through the entire pair list
    for seq1,seq2 in pair_list:
        if i < 10: #Test code only
            i += 1 #Test code only
            seq1, seq2 = single_line_fasta_reader(seq1), single_line_fasta_reader(seq2) #Read each sequence name
            #print('seq 1 looks like', seq1) #Helper code only
            #print('seq 2 looks like', seq2) #Helper code only
            max_scores = create_score_matrix(seq1, seq2, input_matrix, gap_start, gap_extension, normalize)[2]
            #The max score is the third entry of this create_score_matrix return
            list_of_scores.append(max_scores)
            #Helper progress comments
            print('Progress is at', len(list_of_scores)/len(pair_list) * 100, '%')

    return list_of_scores

#Function that generates a list of maximum scores for all items in the pair list fed in.
#Returns a DICTIONARY with keys being gap parameters, values being the corresponding
#positive and negative scores.

def explore_gap_parameters (pos_pair_list, neg_pair_list, gap_starts, gap_extensions, input_matrix):
    #Calculate total progress time
    progress_max = len(gap_starts) * len(gap_extensions)
    #Add to initialized dictionary
    gap_dict = {}
    for i in gap_starts:
        for j in gap_extensions:
            #Call score_generator function to iterate over pair list and return corresponding list of scores.
            pos_list, neg_list = score_generator(pos_pair_list, input_matrix, i, j), score_generator(neg_pair_list, input_matrix, i, j)
            #Create a dataframe with two columns, corresponding to pos and neg pair list.
            combined_df = pd.DataFrame (data = {'Pos':pos_list, 'Neg': neg_list})
            #Create dictionary with first entry being gap start, second entry being gap extensions
            gap_dict[(i, j)] = combined_df
            #Helper progress comments
            print('Dictionary is', len(gap_dict)/progress_max * 100, '% done')

    print('the gap dictionary looks like', gap_dict)
    return gap_dict

#Worker function that returns true positive and false positive rates under three
#conditions: when, or when not, calculating roc curves; and when optimizing or not.

def tp_or_fp_search (dict = None, key = None, threshold = 0.3, roc = False, opt = False): #The 0.3 refers to having 70% TP call rate above this threshold

    #If optimizing, I only feed in a single dataframe. Else, I am iterating over
    #keys (e.g. in the case of gap dictionary)
    if not opt:
        df =  dict[key]
    else:
        df = dict

    #List format to be able to call np.quantile
    pos_column = df['Pos'].tolist()
    neg_column = df['Neg'].tolist()

    #If not calculating roc/optimizing: Calculate false positive rate at a true pos
    #rate of 0.7 by calling np.quantile, which essentially does this for us.
    if not roc and not opt:
        #Calculate the true positive threshold of 0.7 for each key
        #Call np quantile to get the data point threshold
        pos_threshold = np.quantile(pos_column, threshold)

        #Return the number of corresponding false positive pairs above the positive threshold
        number_neg_pairs_above = sum([1 for i in neg_column if i > pos_threshold])

        #The false positive rate is the number of negative pairs' max scores above the positive threshold
        fp_rate = number_neg_pairs_above/len(neg_column)
        print('fp_rate of negatives is for quantile', threshold, 'is', fp_rate)

        #Reassign the dictionary key to be the false positive rate
        return fp_rate

    #If I'm optimizing a scoring matrix: the objective function requires us to look
    #at 4 points, from 0-0.3. So for each of these false positive thresholds, call
    #np.quantile to get the data point that gives corresponding false positive threshold,
    #manually sum up how many positive pair scores are above this.
    elif not roc and opt:
        #Calculate the false positive threshold of range(0, 0.3)
        #Call np quantile to get the data threshold
        false_threshold = np.quantile(neg_column, threshold)

        #Return the number of corresponding true positive pairs above the negative threshold
        number_pos_pairs_above = sum([1 for i in pos_column if i > false_threshold])

        #Similar logic as above
        tp_rate = number_pos_pairs_above/len(pos_column)

        print('tp_rate of positives is for threshold', false_threshold, 'is', tp_rate)
        return tp_rate

    #Handle logic for roc curve calculations; here, threshold isn't a quantile,
    #but is in fact 'cuts' thru the max and min scores of pooled values of neg, pos
    #pair max scores.
    else:
        number_neg_pairs_above = sum([1 for i in neg_column if i > threshold])
        number_pos_pairs_above = sum([1 for i in pos_column if i > threshold])
        #Calculate true and false positives for each cut fed in
        fp_rate = number_neg_pairs_above/len(neg_column)
        tp_rate = number_pos_pairs_above/len(pos_column)
        #Return both rates as data points on roc curve
        return fp_rate, tp_rate

#Wrapper function that generates a combined pos pair and neg pair dataframe for
#a given score matrix and gap start/extension combination; and returns it.
def return_score_df (neg_pairs, pos_pairs, score_matrix, gap_start, gap_extension, normalize = False):
    neg_scores = score_generator(neg_pairs, score_matrix, gap_start = gap_start, gap_extension = gap_extension, normalize = normalize)
    pos_scores = score_generator(pos_pairs, score_matrix, gap_start = gap_start, gap_extension = gap_extension, normalize = normalize)
    #Col names are important here; do not change them or else will run into bugs
    #with tp_or_fp_search function
    combined_df = pd.DataFrame (data = {'Pos': pos_scores, 'Neg': neg_scores})
    return combined_df

#Return roc-curve-ready dataframe; for each cut in the range fed in, returns the
#corresponding true and false positive rates, for each key in the fed in (gap-parameter) dictionary.
#Then, returns the corresponding FP + TP dataframe for each key.

def return_roc_df (dict, range_dict):

    key_values = dict.keys()
    new_dict = {keys: [] for keys in key_values}
    #Get a df for each key
    for key in dict:
        #Actually calculate the FP, TP dataframe by calculating FP, TP for each cut in
        #the range
        for cut in range_dict[key]:
            tuple_fp_tp = tuple(tp_or_fp_search(dict, key, cut, roc = True))
            new_dict[key].append(tuple_fp_tp)
        new_dict[key] = pd.DataFrame(new_dict[key], columns = ['FP', 'TP'])

    print('ROC FP, TP dict is like', new_dict)
    return new_dict

#Plots and saves figure for finding optimal gap penalty combination, by
#plotting false positive rate at TP of 0.7 for each gap penalty combination explored.

def gap_plot (keys, values, name = 'BMI_203_W2020_FP_by_gap_penalty.png', title = 'False positive rate by gap penalty combination' ):
    plt.plot(keys, values)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylim((0, 1))
    plt.ylabel('False positive rate')
    plt.savefig(name)
    plt.clf()

#Plots and saves roc curve for a given roc dataframe as defined from above return_roc_df function.
#Plots FP, TP column.

def roc (dict_roc, fig_name = 'BMI_203_W2020_ROC_scoring_matrices.png', title = 'ROC curves for 5 scoring matrices'):
    colnames = list(dict_roc)
    print('colnames are', colnames)
    #For each scoring matrix that was used with the optimal gap penalty: plot out
    #the corresponding FP and TP points as a line
    for name in colnames:
        plt.plot(dict_roc[name]['FP'], dict_roc[name]['TP'], label = name)
    #Add titles and x, y labels
    plt.title(title)
    plt.ylabel('TP rate')
    plt.xlabel('FP rate')
    #Show the legend, save the figure, and clear the figure afterwards
    plt.legend()
    plt.savefig(fig_name)
    plt.clf()

#---------END PART 1: FUNCTIONS TO FIND OPTIMAL GAP COMBO/ SCORE MATRIX ---------#


#---------BEGIN PART 2: FUNCTIONS FOR OPTIMIZING SCORING MATRIX ---------#

#Helper function to actualize probability sampling

def sample (prob):
    #Simple implementation of probability of an event
    if np.random.uniform() < prob:
        return True
    return False

#Initialize a 'population of matrices' starting from a beginning matrix. How
#this will be done: I will randomly mutate entries in a given matrix at a specified
#'initial probability' (set higher than the mutation probablility in order to search
#a larger space.)

def initialize (matrix, init_prob, number = 4):
    #The new population will be a list of score matrices
    population = []
    #Populate up to the specified number of individuals
    for i in range (number):
        #Deep copy so that the original matrix is unaffected and can be sequentially
        #passed in
        new_matrix = matrix.copy(deep = True)
        #Call below 'mutation' function that with the given probability here, mutates
        #each cell to be another value.
        population.append(mutation(new_matrix, init_prob))

    #print ('The initial population looks like', population) #Test only
    return population

#This function calculates fitness for each matrix that is fed in in the 'part_2.py'
#script. It calculates true positive rates for each false positive threshold and
#then sums these up for the false positive range (0, 0.3). This objective function
#value is then returned. It ranges from [0, 4]. For each matrix, 4 runs through
#the alignment logic occur.

def assess_fitness (neg_pairs, pos_pairs, score_matrix, gap_start, gap_extension, fp_range, normalize = False):
    #Will have 4 values in it
    scores = []
    #i goes from 0, 0.1, 0.2, 0.3
    for i in fp_range:
        #These parameters were defined in part_2.py and are fed in here
        df_for_score_matrix = return_score_df (neg_pairs, pos_pairs, score_matrix, gap_start, gap_extension, normalize)
        #Calculate true positive rate at a threshold of the specified false positive rate
        tp_rate = tp_or_fp_search (df_for_score_matrix, threshold = i, opt = True)
        #Append to the growing scores list
        scores.append(tp_rate)

    #Sum up the list; ranges from 0.0-4.0
    obj_value = sum(scores)
    return obj_value

#Selection works in my algorithm by calculating the median objective function score
#for each of the individuals that are present in my population. I then return
#the individuals that are above this threshold.

def selection (list):
    #Scores are in the second entry of each list element, as defined in part_2.py.
    scores = [i[1] for i in list]
    #Calculate a threshold
    threshold = statistics.median_low(scores)
    #Should get rid of half of the data or less
    selected_popn = [i for i in list if i[1] >= threshold]
    indiv, scores = [i[0] for i in selected_popn], [i[1] for i in selected_popn]
    #Print out the selected individuals left over
    print('The selected population looks like', indiv)
    #The first entry is the actual dataframe corresponding to scores, second is their associated scores
    return indiv, scores

#Genetic algorithms work by recombining fit individuals and mutating them slightly
#with each generation. That logic is imperatively implemented here: the new population,
#given a list of individuals, is a mix of fit individuals, slightly mutated.
def repopulate (df_list, prob = 0.02):
    #Call recombination procedure; need at least 2 in a generation to be a generation!
    new_populations = recombine(df_list, max(len(df_list), 2)) #Might be redundant though
    #Choose a mutation frequency of 0.02 to mimic low frequency of mutation IRL
    #Mutate each new individual
    new_populations = [mutation(i, prob) for i in new_populations]

    return new_populations

#Mutation function takes in a 'score matrix' and mutates each entry with probability 'prob'
def mutation (matrix, prob):
    #Randomly mutate within the maximal and minimal range of the matrix
    min_val = matrix.values.min()
    max_val = matrix.values.max()
    #Matrix is square so this is sufficient
    sizes = matrix.shape[0]

    #Sequentially iterate through the entire dataframe and with a probability 'prob'
    #reassign each entry with a random integer, ranging from minimum value to maximum value
    #of the matrix.
    for i in range(sizes):
        for j in range(sizes):
            if sample (prob): #FALSE or TRUE only
                #Use randint to return a single scalar in this range
                new_value = np.random.randint (min_val, max_val)
                #Symmetry is maintained by this dual assignment
                matrix.iloc[i, j], matrix.iloc[j, i] = new_value, new_value
            assert matrix.iloc[i, j] == matrix.iloc[j, i], 'Symmetry must be maintained.'

    return matrix

#Returns a list of new daughters taken from the list of 'ideal' fit individuals,
#defined above from 'selection' procedure

def recombine (df_list, number):
    new_populations = []
    #number is the number of new individuals going into new generation
    for i in range(number):
        #Only two parents give rise to one daughter df
        two_dfs = random.sample (df_list, 2)
        #Mutate one of the two df's with probability 0.5 to get a daughter
        first_df, second_df = two_dfs[0], two_dfs[1]
        sizes = first_df.shape[0]
        #arbitrary; just select one of two parents to have this work
        daughter_df = first_df.copy(deep = True)
        for i in range(sizes):
            for j in range (sizes):
                if sample(0.5): #0.5 to mirror law of independence of recombination
                    #Multiple assignments to maintain symmetry
                    daughter_df.iloc[i, j], daughter_df.iloc[j, i] = second_df.iloc[i, j], second_df.iloc[j, i]
                assert daughter_df.iloc[i, j] == daughter_df.iloc[j, i], 'Symmetry must be maintained.'
        new_populations.append(daughter_df)

    return new_populations


#---------END PART 2: FUNCTIONS FOR OPTIMIZING SCORING MATRIX ---------#
