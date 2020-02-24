import pandas as pd

def single_line_fasta_reader (fasta):
    with open(fasta) as f:
        seq1 = f.readlines()[1:]
        seq1 = [i.strip() for i in seq1]
        seq1 = ''.join(seq1)

    return seq1

#Expects input matrix format to match that of the given matrix sequences.
def BLOSUM_reader (BLOSUM):
    with open(BLOSUM) as f:
        split_file = f.readlines()
        matrix_name = split_file[0]
        print('The name of the matrix is', matrix_name)

        #Simply remove all lines where '#' is present; represents comment lines. A string is an iterable so use 'not in' construct
        matrix= [i for i in split_file if '#' not in i]

        #Split gets rid of trailing whitespace in the amino acid residue names

        #The aa names are in the first row
        aa_names = matrix[0]
        aa_names = [i for i in aa_names.split()]
        #Strip to get rid of the \n characters at end of string
        numeric_indices = [i.strip() for i in matrix[1:]]

        #Get correct format for each row (important) in the BLOSUM matrix
        for index, scores in enumerate(numeric_indices):
            scores = [int(i) for i in scores.split()]
            numeric_indices[index] = scores

            assert (len(scores) == len(aa_names), 'The length of each BLOSUM score row should match the length of the amino acid residue names.')

    BLOSUM_df = pd.DataFrame(numeric_indices, columns = aa_names, index = aa_names)
    return BLOSUM_df
