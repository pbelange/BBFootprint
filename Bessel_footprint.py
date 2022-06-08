import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import time
from rich.progress import track
from rich.progress import Progress
from rich.progress import Progress, BarColumn, TextColumn,TimeElapsedColumn,SpinnerColumn, TimeRemainingColumn



import Backend.InteractionPoint as inp
import Backend.Detuning as dtune
import Backend.Footprint as fp
import Backend.BeamPhysics as BP

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





# Generating Coord grid
#=========================================================
#coordinates = fp.generate_coordGrid([0.05,10],[0.01*np.pi/2,0.99*np.pi/2],labels = ['r_n','theta_n'],nPoints=500)
coordinates = fp.generate_coordGrid(np.logspace(np.log10(0.5),np.log10(10),10),
                                    np.linspace(0.01*np.pi/2,0.99*np.pi/2,7),labels = ['r_n','theta_n'])

coordinates.insert(0,'x_n',coordinates['r_n']*np.cos(coordinates['theta_n']))
coordinates.insert(1,'y_n',coordinates['r_n']*np.sin(coordinates['theta_n']))


coordinates.insert(0,'J_x',(coordinates['x_n']**2)*B1.emittx/2)
coordinates.insert(1,'J_y',(coordinates['y_n']**2)*B1.emitty/2)

#coordinates.sort_values(by=['r_n'],inplace=True)
#=========================================================




def compute_lr_ho_footprint(coord):
    
    DQx_oct,DQy_oct = np.zeros(len(coord['J_x'])),np.zeros(len(coord['J_x']))
    DQx_ho ,DQy_ho  = np.zeros(len(coord['J_x'])),np.zeros(len(coord['J_x']))

    with Progress(
    "{task.description}",
    SpinnerColumn(),
    BarColumn(bar_width=40),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),TimeElapsedColumn(),TimeRemainingColumn()) as progress:

        task1 = progress.add_task("[red]Both IPs...", total=2)
        task2 = progress.add_task("[green]Each BBLR...", total=len(IP1.lr))
        task3 = progress.add_task("[green]Remaining time", total=len(IP1.lr))

        for _IP in [IP1,IP5]:

            # Computing octupolar effect
            #------------------------------
            # Iterating over LR only
            progress.update(task2,completed=0,refresh =True)
            progress.update(task3,completed=0,refresh =True)
            for index, _bb in _IP.lr.iterrows():
                _DQx,_DQy = dtune.DQx_DQy( ax   = coord['x_n'],
                                           ay   = coord['y_n'],
                                           r    = _bb['r'],
                                           dx_n = _bb['dx_n'],
                                           dy_n = _bb['dy_n'],
                                           xi   = _IP.b2.xi)

                DQx_oct += _DQx
                DQy_oct += _DQy
                
                progress.reset(task2,completed = progress.tasks[task2].completed)
                progress.update(task2,advance=1,refresh =True)
                progress.update(task3,advance=1,refresh =True)


            # Computing Head-on component (ex: bb_ho.c1b1_00):
            main_ho = _IP.ho.loc[f'bb_ho.c{_IP.name[-1]}b1_00']
            #------------------------------
            _DQx,_DQy = dtune.DQx_DQy( ax   = coord['x_n'],
                                       ay   = coord['y_n'],
                                       r    = main_ho['r'],
                                       dx_n = main_ho['dx_n'],
                                       dy_n = main_ho['dy_n'],
                                       xi   = _IP.b2.xi)
            DQx_ho += _DQx
            DQy_ho += _DQy
            
            progress.update(task1,advance=1,refresh =True)
        
    return (DQx_oct,DQy_oct),(DQx_ho,DQy_ho)


# COMPUTING
#=========================
s_time = time.time()
(DQx_lr,DQy_lr),(DQx_ho,DQy_ho) = compute_lr_ho_footprint(coordinates)
e_time = time.time()


# Saving data
coordinates.insert(1,'DQx_lr',DQx_lr)
coordinates.insert(2,'DQy_lr',DQy_lr)
coordinates.insert(3,'DQx_ho',DQx_ho)
coordinates.insert(4,'DQy_ho',DQy_ho)

coordinates.to_pickle(f"Results/Bessel_footprint.pkl")
 
    
    
    

# PLOTTING
#================================================================
Qx_0,Qy_0 = 0.31,0.32

cmap = 'viridis'
fig,axes = plt.subplots(2,2,figsize=(12,8))
fig.suptitle(f'Execution time: {(e_time-s_time):.3f} s')

# Plotting coordinates
#-----------------------
plt.sca(axes[0,0])
axes[0,0].set_title('Coordinates')
plt.scatter(coordinates['x_n'],coordinates['y_n'],s=5,c=coordinates['r_n'],alpha=0.8,norm =BoundaryNorm(boundaries=np.linspace(0,10,11), ncolors=int(0.9*256)))
plt.xlabel(r'$x_n$',fontsize=16);
plt.ylabel(r'$y_n$',fontsize=16);
plt.axis('equal');
cbar = plt.colorbar()
plt.set_cmap(cmap)
cbar.ax.set_ylim([0,np.max(coordinates['r_n'])])
cbar.ax.set_ylabel(r'$\sqrt{(2J_x + 2J_y)/\varepsilon}$ [$\sigma$]',fontsize=12)

# Plotting LR only
#-----------------------
plt.sca(axes[0,1])
axes[0,1].set_title('LR only')
BP.plotWorkingDiagram(order = 12,QxRange=np.array([0.25,0.35]),QyRange=np.array([0.25,0.35]),alpha=0.2,zorder=-1000)
plt.plot([Qx_0],[Qy_0],'P',markersize=5,color='C3',alpha=0.5,label='Unperturbed W.P.')

plt.scatter(Qx_0 + coordinates['DQx_lr'],Qy_0 + coordinates['DQy_lr'],s=5,c=coordinates['r_n'],alpha=0.8,norm =BoundaryNorm(boundaries=np.linspace(0,10,11), ncolors=int(0.9*256)))
plt.xlabel(r'$Q_x$',fontsize=16);
plt.ylabel(r'$Q_y$',fontsize=16);
plt.axis('equal');
window = 0.02
plt.xlim([0.31-window,0.31+window])
plt.ylim([0.32-window,0.32+window])
cbar = plt.colorbar()
plt.set_cmap(cmap)
cbar.ax.set_ylim([0,np.max(coordinates['r_n'])])
cbar.ax.set_ylabel(r'$\sqrt{(2J_x + 2J_y)/\varepsilon}$ [$\sigma$]',fontsize=12)

# Plotting head-on only
#-----------------------
plt.sca(axes[1,0])
axes[1,0].set_title('Head-on only')
BP.plotWorkingDiagram(order = 12,QxRange=np.array([0.25,0.35]),QyRange=np.array([0.25,0.35]),alpha=0.2,zorder=-1000)
plt.plot([Qx_0],[Qy_0],'P',markersize=5,color='C3',alpha=0.5,label='Unperturbed W.P.')

plt.scatter(Qx_0 + coordinates['DQx_ho'],Qy_0 + coordinates['DQy_ho'],s=5,c=coordinates['r_n'],alpha=0.8,norm =BoundaryNorm(boundaries=np.linspace(0,10,11), ncolors=int(0.9*256)))
plt.xlabel(r'$Q_x$',fontsize=16);
plt.ylabel(r'$Q_y$',fontsize=16);
plt.axis('equal');
window = 0.02
plt.xlim([0.31-window,0.31+window])
plt.ylim([0.32-window,0.32+window])
cbar = plt.colorbar()
plt.set_cmap(cmap)
cbar.ax.set_ylim([0,np.max(coordinates['r_n'])])
cbar.ax.set_ylabel(r'$\sqrt{(2J_x + 2J_y)/\varepsilon}$ [$\sigma$]',fontsize=12)


# Plotting all together
#-----------------------
plt.sca(axes[1,1])
axes[1,1].set_title('HO + LR')
BP.plotWorkingDiagram(order = 12,QxRange=np.array([0.25,0.35]),QyRange=np.array([0.25,0.35]),alpha=0.2,zorder=-1000)
plt.plot([Qx_0],[Qy_0],'P',markersize=5,color='C3',alpha=0.5,label='Unperturbed W.P.')

plt.scatter(Qx_0 + coordinates['DQx_lr']+ coordinates['DQx_ho'],Qy_0 + coordinates['DQy_lr'] + coordinates['DQy_ho'],s=5,c=coordinates['r_n'],alpha=0.8,norm =BoundaryNorm(boundaries=np.linspace(0,10,11), ncolors=int(0.9*256)))
plt.xlabel(r'$Q_x$',fontsize=16);
plt.ylabel(r'$Q_y$',fontsize=16);
plt.axis('equal');
window = 0.02
plt.xlim([0.31-window,0.31+window])
plt.ylim([0.32-window,0.32+window])
cbar = plt.colorbar()
plt.set_cmap(cmap)
cbar.ax.set_ylim([0,np.max(coordinates['r_n'])])
cbar.ax.set_ylabel(r'$\sqrt{(2J_x + 2J_y)/\varepsilon}$ [$\sigma$]',fontsize=12)

plt.tight_layout()
plt.savefig('Results/Bessel_footprint.pdf',format='pdf')

