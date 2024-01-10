a = 'antigen_a,antibody_b'
antigen = a.split(',')[0]
antibody = a.split(',')[1]
antigen_pid = antigen.split('_')[0]
antigen_ch = antigen.split('_')[1]
antibody_pid = antibody.split('_')[0]
antibody_ch = antibody.split('_')[1]
print(antigen)
print(antibody)
print(antigen_pid)
print(antibody_pid)
print(antigen_ch)
print(antibody_ch)