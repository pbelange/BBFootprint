#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed March 1 11:50:29 2021

@author: pbelange
"""

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.collections
import numpy as np
import pandas as pd
import PyNAFF as pnf

     
#==========================================================================================
# NOTES:
#------------------------------------------------------------------------------------------
#-> Proposing to have a transparent constant library, instead of using scipy. Maybe import from MadX?
#-> this line should be done in the mad class ... twiss_dfs={'lhcb1':mad.get_twiss_df('lhcb1_twiss'),'lhcb2':mad.get_twiss_df('lhcb2_twiss')}
#-> Should I make the format of madx call tidier, so that the log is easily readable? 
#==========================================================================================

#==========================================================================================
# MAD-X TEMPLATE
#------------------------------------------------------------------------------------------
"""
'''
    option,trace;
    small=0.05;
    big=sqrt(1.-small^2);
    track,dump, onetable = true;
    xs=small; ys=small;
    value,xs,ys;
    start,fx=xs,fy=ys;  // zero amplitude
    nsigmax=6;
    n=1; // sigma multiplier
    m=0; // angle multiplier
    while (n <= nsigmax)
    {
      angle = 15*m*pi/180;
      if (m == 0) {xs=n*big; ys=n*small;}
      elseif (m == 6) {xs=n*small; ys=n*big;}
      else
      {
        xs=n*cos(angle);
        ys=n*sin(angle);
      }
      value,xs,ys;
      start,fx=xs,fy=ys;
      m=m+1;
      if (m == 7) { m=0; n=n+1;}
    };
    ! you should run only not dynap
    run,turns=1024;
    dynap,fastune,turns=1024;
    endtrack;  
    '''
"""
#==========================================================================================


def generate_coordGrid(xRange,yRange,labels = ['x','y'],nPoints=100):
    '''
    Distribute points uniformly on a 2D grid.
    -----------------------------------------
    Input:
        xRange : range of first coordinate
        yRange : range of second coordinate
        labels : labels to be used in the resulting dataframe
        nPoint : total number of points to generate (sqrt(nPoints) for each coordinate)
    Returns:
        coordinates: dataframe containing the distributed points
    '''

    if type(xRange) is list and type(yRange) is list:
        xVec = np.linspace(xRange[0],xRange[1],int(np.sqrt(nPoints)))
        yVec = np.linspace(yRange[0],yRange[1],int(np.sqrt(nPoints)))
    else:
        xVec = xRange
        yVec = yRange
        
    xx,yy = np.meshgrid(xVec,yVec)
    xx,yy = xx.flatten(),yy.flatten()

    return pd.DataFrame(dict(zip(labels,[xx,yy])))




def getTune_pynaff(particle,nterms = 1,skipTurns=0):
    '''
    Compute tunes of a particle using pyNAFF
    ----------------------------------------
    Input:
        particle: pd.Series or dict containing x(t) and y(t) of the particle
        nterms : maximum number of harmonics to search for in the data sample
        skipTurns : number of observations (data points) to skip from the start
    Returns:
        tunes: pd.Series containing the tune in both planes
    '''

    NAFF_X = pnf.naff(np.array(particle['x'])-np.mean(particle['x']), turns=len(particle), nterms=nterms , skipTurns=skipTurns, getFullSpectrum=False)
    NAFF_Y = pnf.naff(np.array(particle['y'])-np.mean(particle['y']), turns=len(particle), nterms=nterms , skipTurns=skipTurns, getFullSpectrum=False)
    
    # TODO: allow more than 1 harmonic (nterms>1)
    # naff returns: [order of harmonic, frequency, Amplitude, Re{Amplitude}, Im{Amplitude] 
    _,Qx,_,Ax_Re,Ax_Im = NAFF_X[0]
    _,Qy,_,Ay_Re,Ay_Im = NAFF_Y[0]    
    
    return pd.Series({'Qx':Qx,'Qy':Qy})


def getTune_fft (particle,showSpectrum=False):
    '''
    Compute tunes of a particle from simple FFT
    -------------------------------------------
    Input:
        particle: pd.Series or dict containing x(t) and y(t) of the particle
        showSpectrum: {True|False} to plot the spectrum used for the fft
    Returns:
        tunes: pd.Series containing the tune in both planes
    '''

    turns = np.arange(1,len(particle['x'])+1)
    freq = np.fft.fftfreq(turns.shape[-1])

    tunes = {'Qx':0,'Qy':0}
    for plane in ['x','y']:
        spectrum = np.fft.fft(particle[plane]-np.mean(particle[plane]))
        idx = np.argmax(np.abs(spectrum))
        tunes['Q'+plane] = freq[idx]


    # For debuging purposes    
    if showSpectrum:
        plt.figure()
        plt.plot(freq,np.abs(np.fft.fft(particle['x']-np.mean(particle['x']))),label='FFT(x(t))')
        plt.plot(freq,np.abs(np.fft.fft(particle['y']-np.mean(particle['y']))),label='FFT(y(t))')
        plt.xlim([0,np.max(freq)])  
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    return pd.Series(tunes)

  

def compute_tunes(tracked,method='pynaff'):
    '''
    Apply the chosen method accross all the particles in a dataframe
    ----------------------------------------------------------------
    Input:
        tracked: pd.DataFrame from madx trackone table
    Returns:
        tuneDF : pd.DataFrame containing the tune of all tracked particles
    '''

    getTune = { 'pynaff':getTune_pynaff,
                'fft'   :getTune_fft     }[method.lower()]
    
    return tracked.groupby('number').apply(getTune)



def draw_footprint(A, axis_object=None, figure_object=None, axis=0, linewidth=4):
    '''
    Input A should be a 3-D numpy array with shape (Nx,Ny,2)
    representing a 2-D array of (x,y) points. This function
    will draw lines between adjacent points in the 2-D array.
    '''
    if len(A.shape) != 3:
        print('ERROR: Invalid input matrix')
        return None
    if A.shape[2] != 2:
        print('ERROR: Points are not defined in 2D space')
        return None

    sx = A.shape[0]-1
    sy = A.shape[1]-1

    p1 = A[:-1,:-1,:].reshape(sx*sy,2)[:,:]
    p2 = A[1:,:-1,:].reshape(sx*sy,2)[:]
    p3 = A[1:,1:,:].reshape(sx*sy,2)[:]
    p4 = A[:-1,1:,:].reshape(sx*sy,2)[:]

    #Stack endpoints to form polygons
    Polygons = np.stack((p1,p2,p3,p4))
    #transpose polygons
    Polygons = np.transpose(Polygons,(1,0,2))
    patches = list(map(matplotlib.patches.Polygon,Polygons))

    #assign colors
    patch_colors = [(0,0,0) for a in Polygons]
    #patch_colors[(sx-1)*sy:] = [(0,1,0)]*sy
    #patch_colors[(sy-1)::sy] = [(0,0,1)]*sx

    p_collection = matplotlib.collections.PatchCollection(patches,facecolors=[],linewidth=linewidth,edgecolor=patch_colors)

    if axis_object is None:
        if figure_object:
            fig = figure_object
        else:
            fig = plt.figure()
        if len(fig.axes) == 0:
            plt.subplot(1,1,1)
        if axis >= len(fig.axes) or axis < 0:
            i = 0
        else:
            i = axis
        ax = fig.axes[i]
    else:
        ax = axis_object
        fig = None

    ax.add_collection(p_collection)

    return fig