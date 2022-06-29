import subprocess
import sys
import os




# Running pymask
cwd = os.getcwd()
os.chdir('/home/pbelange/abp/Apps/lhcmask/python_examples/run3_collisions_wire')
exec(open("000_pymask_rich.py").read())
opticsFile = configuration['optics_file'].split('/')[-1]
os.chdir(cwd)

# NOTE: Make sure you add: 
# pm.install_lenses_in_sequence(mad_track, bb_dfs['b2'], 'lhcb2')
# at line 438 of '000_pymask_rich.py'

# Saving sequences and BB dfs
for seq in ['lhcb1','lhcb2']:
    mad_track.input(f'use, sequence={seq};')
    mad_track.twiss()
    mad_track.survey()
    
    twiss = mad_track.table.twiss.dframe()
    survey = mad_track.table.survey.dframe()

    twiss.to_pickle(f"LHC_sequence/{seq}_opticsfile{opticsFile.split('.')[-1]}_twiss.pkl")
    survey.to_pickle(f"LHC_sequence/{seq}_opticsfile{opticsFile.split('.')[-1]}_survey.pkl")
    
bb_dfs['b1'].to_pickle(f"LHC_sequence/lhcb1_opticsfile{opticsFile.split('.')[-1]}_bb_dfs.pkl")
bb_dfs['b2'].to_pickle(f"LHC_sequence/lhcb2_opticsfile{opticsFile.split('.')[-1]}_bb_dfs.pkl")