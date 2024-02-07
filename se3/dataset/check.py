input_file = open('bep3/24/7a0y_B.pdb')

letters = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H',
           'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
           'TYR': 'Y', 'VAL': 'V'}
name = '7a0y_B'
name = '>' + name;
print(name)
# print '>',name[0:len(name)]
prev = '-1'
seq = []
for line in input_file:
    toks = line.split()
    if len(toks) < 1: continue
    if toks[0] != 'ATOM': continue
    if toks[5] != prev:
        seq.extend(letters[toks[3]])
    prev = toks[5]
print(seq)



