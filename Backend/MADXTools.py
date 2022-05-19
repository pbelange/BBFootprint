import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import matplotlib.transforms as transforms
import scipy.special as sciSpec
import subprocess
import tfs

import Backend.Constants as cst
import Backend.WCTools as WCTools


##############################################################
markerDict = {'dipedge':'s',
              'drift':'|', 
              'hkicker':'>', 
              'instrument':2, 
              'marker':'X', 
              'monitor':4, 
              'multipole':'*', 
              'octupole':8, 
              'placeholder':'_', 
              'quadrupole':'D', 
              'rbend':'s', 
              'rcollimator':7, 
              'rfcavity':'P', 
              'sbend':'s', 
              'sextupole':'h', 
              'solenoid':6, 
              'tkicker':',', 
              'vkicker':'^'}

colorDict = dict.fromkeys(markerDict.keys(),'k')
colorDict.update({'sbend':'C0','rbend':'C0','quadrupole':'C1','rfcavity':'C3','sextupole': 'C2'})


ARCSNum = [''.join(list(np.roll(range(1,9),-i)[:2].astype(str))) for i in range(0,8)]
##############################################################



##############################################################
def get_tracking_string(particleData):
    '''
    Creates tracking string from particle initial coordinates.
    ----------------------------------------------------------
    Input:
        particleData: pd.Series/dict containing canonical or action-angle coordinates
    Output:
        trackingCmd: formated string "start,x=0,px=0,y=0,py=0,t=0,pt=0,fx=0,phix=0,fy=0,phiy=0,ft=0,phit=0;"
        (Default value is 0 for all coordinates)
    '''

    coordinates = ['x','px','y' ,'py' ,'t' ,'pt', 'fx','phix','fy','phiy','ft','phit']
    trackingCmd = ['start']
    trackingCmd += [f'{coord}={particleData.get(coord,0)}' for coord in coordinates]
    trackingCmd = ','.join(trackingCmd)
    return trackingCmd
##############################################################


##############################################################
def MADTrackParticles(coordinates,NTurns = 1,saveFile = None,onepass='onepass'):

    trackingCmds = coordinates.apply(get_tracking_string,axis=1)
    trackingCmds = ';\n'.join(trackingCmds)
    
    saveCmd = ''
    if saveFile is not None:
        saveCmd = f'WRITE, TABLE=trackone, FILE={saveFile}'
    
    madCall = ( f"track,dump,{onepass}, onetable = true,file=trackone.trk;\n"
                f"\n"
                f"!{40*'-'}\n"
                f"{trackingCmds};\n"
                f"!{40*'-'}\n"
                f"\n"
                f"run,turns={NTurns};\n"
                f"endtrack;\n"
                f"{saveCmd};")

    return madCall
##############################################################


##############################################################
##############################################################
def seqedit(sequence,editing, makeThin = False):
    
    

    output = ''
    if makeThin:
        output = f'''
        use, sequence = {sequence};
        makethin,sequence = {sequence};'''
    
    # install,element = multipole_wire_1of2,class=_multipole_wire_1of2 ,at = 0.5
    # PREVIOUS VERSION
    #elementsEntry = '\n'.join([f'{row["mode"]},element = {row["name"]},class={row["definition"].split(":")[0]},at = {row["at"]};' for _,row in editing.iterrows()])
    #definitionEntry = '\n'.join(editing['definition'])
    #-----------                       
    
    
    
    # SORTING
    if 'at' in list(editing.columns):
        editing.sort_values('at',inplace=True)
    
    # DEFINITION FOR INSTALLATION
    if 'definition' in list(editing.columns):
        definitionEntry = '\n'.join(editing['definition'])
    else: 
        definitionEntry = ''
        
    
    
    
    # ELEMENTS
    def installStr(row):
        return f'{row["mode"]},element = {row["name"]},class={row["definition"].split(":")[0]},at = {row["at"]};' 

    def removeStr(row):
        return f'{row["mode"]},element = {row["name"]};' 
    
    def skipStr(row):
        return ''

    entryStr = {'install':installStr,'remove':removeStr,'skip':skipStr}

    elementsEntry = '\n'.join(filter(None, [entryStr[row['mode']](row) for _,row in editing.iterrows()]))
    
    

    output += f'''
    
        {definitionEntry}
    
        use, sequence = {sequence};
        SEQEDIT, SEQUENCE={sequence};
            FLATTEN;
            {elementsEntry}
            FLATTEN;
        ENDEDIT;

        use, sequence = {sequence};
    '''
    
    if makeThin:
        output += f'''
        use, sequence = {sequence};
        makethin,sequence = {sequence};'''
    
                               
    return output
##############################################################
##############################################################
                               
                               
                            
##############################################################
madSetup = '''
!-------------------
! Defining sequence
!-------------------

{name}:sequence, refer = center, L={L_seq};
!------------------------
!------------------------
endsequence;


!-------------------
! Defining Beam 
!-------------------
beam,   particle = proton,
        charge   = 1,
        npart    = 1,
        energy   = {Energy}/1e9;

!-------------------
! Twiss and MakeThin
!-------------------
use, sequence = {name}; 
makethin,sequence = {name};
'''
##############################################################
                               
                               
##############################################################

madMatch = '''
    use, period = {name};
    match;

    vary, name=K_Qf,step=.001,UPPER=5,LOWER=-5;
    vary, name=K_Qd,step=.001,UPPER=5,LOWER=-5;
    
    constraint,range=#end,mux={mux},muy={muy};

    lmdif,calls=100;
    endmatch;


    title, 'Twiss';
    twiss;
'''
##############################################################
                               
                               

##############################################################
def plotElements(twissDF,ax=None):
    if ax is None:
        ax = plt.gca()
        
    ax.plot(twissDF['s'],0*twissDF['s'],'k')

    colors = {'quadrupole':'C3','multipole':'C2','sbend':'C0'}
    alpha = 0.5
    linewidth = 3
    
    # Adding Quadrupoles:
    keyword = 'quadrupole'
    for index,element in twissDF[twissDF['keyword']==keyword].iterrows():
        ax.add_patch(patches.Rectangle(
            (element.s-element.l, 0),   # (x,y)
            element.l,          # width
            element.k1l,          # height
            color=colors[keyword], alpha=alpha,lw=linewidth ))
        
    # Adding Multipoles:
    keyword = 'multipole'
    for index,element in twissDF[twissDF['keyword']==keyword].iterrows():
        ax.add_patch(patches.Rectangle(
            (element.s-element.l, 0),   # (x,y)
            element.l,          # width
            element.k1l,          # height
            color=colors[keyword], alpha=alpha,lw=linewidth ))
        
    # Adding Multipoles:
    keyword = 'sbend'
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for index,element in twissDF[twissDF['keyword']==keyword].iterrows():
        if element.angle !=0:
            height = 1/8
            ax.add_patch(patches.Rectangle(
                (element.s-element.l, 0.5-height/2),   # (x,y)
                element.l,          # width
                height,          # height
                color=colors[keyword], alpha=alpha,lw=linewidth,transform=trans ))


##############################################################


##############################################################
def plotSurvey(surveyTable):
    
    survey = surveyTable.copy()
    survey.loc[:,'x'] -= np.mean(survey['x'])
    r, theta = WCTools.cart2pol(survey['z'],survey['x'])
    survey.insert(0,'r_polar',r)
    survey.insert(0,'rotation',theta)

    IP_loc = survey.loc[[f'ip{IP}' for IP in range(1,9)]]
    _groupedSurvey = survey.groupby('keyword')

    
    plt.plot(survey['z'],survey['x'],'k',alpha=0.3)
    for gr in list(_groupedSurvey.groups.keys()):
        thisAlpha = 0.3 if colorDict[gr] == 'k' else 1
        plt.plot(_groupedSurvey.get_group(gr)['z'],_groupedSurvey.get_group(gr)['x'],alpha=thisAlpha,linestyle='None',color=colorDict[gr],marker=markerDict[gr],label=gr)

    # Adding s axis:
    axScale = 0.9
    plt.plot(axScale*survey['z'],axScale*survey['x'],'k',alpha=1)
    for index, row in IP_loc.iterrows():
        plt.plot(1.01*axScale*row['z'],1.01*axScale*row['x'],'k',ms=10, mew=1,marker=(1,1,90+np.rad2deg(row['rotation'])),linestyle='None',alpha=1)
        plt.text(0.85*axScale*row['z'],0.85*axScale*row['x'],row['name'].upper(),ha='center',va='center',rotation=-90+np.rad2deg(np.mod(row['rotation'],np.pi)))

    # Adding arc delemiter    
    for i in range(0,8):
        arc = f'{np.roll(range(1,9),-i)[0]}{np.roll(range(1,9),-i)[1]}'

        thisArc = survey.loc[f's.arc.{arc}.b1':f'e.arc.{arc}.b1']
        plt.plot(axScale*thisArc['z'],axScale*thisArc['x'],'C0',alpha=0.8,lw=5,zorder=-10)

        plt.text(0.85*axScale*thisArc.iloc[len(thisArc)//2]['z'],0.85*axScale*thisArc.iloc[len(thisArc)//2]['x'],f'ARC{arc}',fontsize=7,ha='center',va='center',rotation=-90+np.rad2deg(np.mod(thisArc.iloc[len(thisArc)//2]['rotation'],np.pi)))

    plt.legend()
    plt.axis('equal')
    plt.axis('off')

    return survey
##############################################################



##############################################################
def plotTwiss(twissTable):
    
    twiss = twissTable.copy()
    _groupedTwiss = twiss.groupby('keyword')
    
    plt.plot(twiss['s'],twiss['betx'],'k',alpha=0.3)
    for gr in list(_groupedTwiss.groups.keys()):
        thisAlpha = 0.3 if colorDict[gr] == 'k' else 1
        plt.plot(_groupedTwiss.get_group(gr)['s'],_groupedTwiss.get_group(gr)['betx'],alpha=thisAlpha,linestyle='None',color=colorDict[gr],marker=markerDict[gr],label=gr)

    plt.legend()

    # Adding arc delimiters:  
    for i in range(0,8):
        arc = f'{np.roll(range(1,9),-i)[0]}{np.roll(range(1,9),-i)[1]}'

        thisArc = twiss.loc[f's.arc.{arc}.b1':f'e.arc.{arc}.b1']
        plt.gca().add_patch(patches.Rectangle((np.min(thisArc['s']),0), 
                                              np.max(thisArc['s'])-np.min(thisArc['s']), 
                                              300,
                                             facecolor='none',edgecolor='k'))


    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.set_xbound(ax1.get_xbound())

    IP_loc = {'s':[],'label':[]}
    for IP in range(1,9):
        IP_loc['s'].append(float(twiss.loc[twiss.name.str.fullmatch(f'ip{IP}:1'),'s']))
        IP_loc['label'].append(f'IP{IP}')

    ax2.set_xticks(IP_loc['s']);
    ax2.set_xticklabels(IP_loc['label']);
##############################################################
