from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track as pbar
import json
import pandas as pd

import Backend.InteractionPoint as inp
import Backend.Detuning as dtune
import Backend.Footprint as fp
import Backend.BeamPhysics as BP
import Backend.Constants as cst

import xobjects as xo
import xtrack as xt
import xpart as xp


# Importing twiss and survey
twiss_b1  = pd.read_pickle('LHC_sequence/lhcb1_twiss.pkl')
survey_b1 = pd.read_pickle('LHC_sequence/lhcb1_survey.pkl')

twiss_b2  = pd.read_pickle('LHC_sequence/lhcb2_twiss.pkl')
survey_b2 = pd.read_pickle('LHC_sequence/lhcb2_survey.pkl')


B1 = inp.Beam('b1',twiss_b1,survey_b1,
              Nb       = 1.15e11,
              E        = 6.8e12,
              emittx_n = 2.5e-6,
              emitty_n = 2.5e-6,
              dp_p0    = 0)
    
B2 = inp.Beam('b2',twiss_b2,survey_b2,
              Nb       = 1.15e11,
              E        = 6.8e12,
              emittx_n = 2.5e-6,
              emitty_n = 2.5e-6,
              dp_p0    = 0)

IP1 = inp.InteractionPoint('ip1',B1,B2)
IP5 = inp.InteractionPoint('ip5',B1,B2)

# Setting up Tracking
#===================================================
beam = 'b1'
fname_line_particles= f'xsuite_lines/001_{beam}_line_bb_wire.json'
with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data)
line.particle_ref = xp.Particles.from_dict(input_data['particle_on_tracker_co'])

tracker = xt.Tracker(line=line)
tw = tracker.twiss()
clear_output(wait=False) #to clear xtrack output



    
# Generating Coord grid
#=========================================================
#coordinates = fp.generate_coordGrid([0.05,10],[0.01*np.pi/2,0.99*np.pi/2],labels = ['r_n','theta_n'],nPoints=500)
coordinates = fp.generate_coordGrid(np.linspace(0.5,10,10),
                                    np.linspace(0.01*np.pi/2,0.99*np.pi/2,3),labels = ['r_n','theta_n'])

coordinates.insert(0,'x_n',coordinates['r_n']*np.cos(coordinates['theta_n']))
coordinates.insert(1,'y_n',coordinates['r_n']*np.sin(coordinates['theta_n']))

coordinates.insert(0,'J_x',(coordinates['x_n']**2)*B1.emittx/2)
coordinates.insert(1,'J_y',(coordinates['y_n']**2)*B1.emitty/2)
 
for plane in ['x','y']:
    alpha,beta = tw[f'alf{plane}'][0],tw[f'bet{plane}'][0]
    gamma = (1+alpha**2)/beta

    phi = 0
    coordinates.insert(0,f'{plane}'  ,np.sqrt(2*beta*coordinates[f'J_{plane}'])*np.cos(phi))
    coordinates.insert(1,f'p_{plane}',-np.sqrt(2*coordinates[f'J_{plane}']/beta)*(np.sin(phi)+alpha*np.cos(phi)))
#========================================================= 


# Deactivating all wires
#===================================================
tracker.vars['enable_qff'] = 0
for IP in ['ip1','ip5']:
    tracker.vars[f"bbcw_rw_{IP}.{beam}"] = 1
    tracker.vars[f"bbcw_i_{IP}.{beam}"]  = 0

    
# Deactivating SEXTUPOLES and  OCTUPOLES
#===================================================
allVars = list(tracker.vars._owner.keys())
allElements = list(tracker.element_refs._owner.keys())

ks = [name for name in allVars if ('ksf' in name)|('ksd' in name)]
ko = [name for name in allVars if ('kof.a' in name)|('kod.a' in name)]

tracker.vars['all_oct_ON']  = 0
tracker.vars['all_sext_ON'] = 0
for _ks in ks:
    tracker.vars[_ks] = tracker.vars['all_sext_ON']*tracker.vars[_ks]._expr    
for _ko in ko:
    tracker.vars[_ko] = tracker.vars['all_oct_ON'] *tracker.vars[_ko]._expr 


# Keeping only LR in ip1 and ip5:
#===================================================

#.N_part_per_slice
for _ip in ['ip1','ip5','ip2','ip8']:
    bb_lr = [name for name in allElements if ('bb_lr' in name)&(f'{_ip[-1]}b1' in name)]
    bb_ho = [name for name in allElements if ('bb_ho' in name)&(f'{_ip[-1]}b1' in name)]

    # New knob:
    tracker.vars[f'{_ip}_bblr_ON'] = 0
    tracker.vars[f'{_ip}_bbho_ON'] = 0
    
    
    # Linking to new knob
    for _lr in bb_lr:
        tracker.element_refs[_lr].n_particles      = tracker.vars[f'{_ip}_bblr_ON']*tracker.element_refs[_lr].n_particles._value
    for _ho in bb_ho:
        tracker.element_refs[_ho].N_part_per_slice = tracker.vars[f'{_ip}_bbho_ON']*tracker.element_refs[_ho].N_part_per_slice._value
        
tracker.vars[f'ip1_bblr_ON'] = 1
tracker.vars[f'ip5_bblr_ON'] = 1
        
# Setting up the tracker
particles = xp.Particles(
                        p0c=B1.p0*cst.c,
                        x =coordinates['x'],
                        px=coordinates['p_x'],
                        y =coordinates['y'],
                        py=coordinates['p_y'],
                        )


print('START TRACKING')
## Track (saving turn-by-turn data)
n_turns = int(1024)
tracker.track(particles, num_turns=n_turns,turn_by_turn_monitor=True)

#================

import pandas as pd

#CONVERT TO PANDAS
tracked = pd.DataFrame(tracker.record_last_track.to_dict()['data'])
tracked = tracked[['at_turn','particle_id','x','px','y','py']]
tracked.rename(columns={"at_turn": "turn",'particle_id':'number'},inplace=True)


