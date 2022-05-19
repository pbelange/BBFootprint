#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022
@author: pbelange
Description :
"""

import numpy as np
import scipy.integrate as integrate

import Backend.Constants as cst



#================================================================================
#    Full analytic detuning
#================================================================================

def DQx_DQy(r,ax,ay,dx,dy,xi):
    DQx = -2*xi*dC00dx(r,ax,ay,dx,dy)/ax
    DQy = -2*xi*dC00dy(r,ax,ay,dx,dy)/ay
    return DQx,DQy

#================================================================================
#---------------------------------------
def g(t,r):
    return np.sqrt(1+(r**2-1)*t)
#---------------------------------------
#---------------------------------------
def Bess2D_generating_real(phi,X,Y,n):
    arg =  - X*np.sin(phi) + 2*Y*np.sin(phi)**2
    return np.cos(n*phi)*np.exp(arg)

def Bess2D_generating_imag(phi,X,Y,n):
    arg =  - X*np.sin(phi) + 2*Y*np.sin(phi)**2
    return -np.sin(n*phi)*np.exp(arg)


def Bess2D(X,Y,n):

    # Choosing between the real part or the imaginary part depending on n
    generatingFun,sign = {0:(Bess2D_generating_real,(-1)**(n/2)),
                          1:(Bess2D_generating_imag,(-1)**((n-1)/2))}[n%2]
    # computing the integral
    integratedFactor = integrate.quad(lambda phi: generatingFun(phi,X,Y,n), 0, 2*np.pi)[0]

    return sign*np.exp(-X-2*Y)/2/np.pi * integratedFactor
#---------------------------------------
#---------------------------------------
def Q0z(t,azbar,dzbar):
    X =  t*azbar*dzbar
    Y = -t*azbar**2/4
    return np.exp(-t/2*(azbar-dzbar)**2)*Bess2D(X,Y,0)
#---------------------------------------
#---------------------------------------
def dQ0daz(t,azbar,dzbar,etaz):
    X    =  t*azbar*dzbar
    Y    = -t*azbar**2/4
    return etaz*np.exp(-t/2*(azbar-dzbar)**2)*(-azbar/2*(Bess2D(X,Y,0)+Bess2D(X,Y,2)) + dzbar*Bess2D(X,Y,1))
#---------------------------------------
#---------------------------------------
def dC00dx_generating(t,r,ax,ay,dx,dy):
    # bar variables
    axbar,aybar,dxbar,dybar = ax*r, ay/g(t,r) , dx , r*dy/g(t,r)

    # modified version of eta to include all the prefactors
    etax_modif = r/g(t,r)

    return dQ0daz(t,axbar,dxbar,etax_modif)*Q0z(t,aybar,dybar)

def dC00dy_generating(t,r,ax,ay,dx,dy):
    # bar variables
    axbar,aybar,dxbar,dybar = ax*r, ay/g(t,r) , dx , r*dy/g(t,r)

    # modified version of eta to include all the prefactors
    etay_modif = 1/(g(t,r)**2)

    return dQ0daz(t,aybar,dybar,etay_modif)*Q0z(t,axbar,dxbar)

def dC00dx(r,ax,ay,dx,dy):
    return integrate.quad(lambda t: dC00dx_generating(t,r,ax,ay,dx,dy), 0, 1)[0]


def dC00dy(r,ax,ay,dx,dy):
    return integrate.quad(lambda t: dC00dy_generating(t,r,ax,ay,dx,dy), 0, 1)[0]
#---------------------------------------
#================================================================================
#================================================================================




#================================================================================
#    Octupolar approximation
#================================================================================

def BBLR_octupole(Jx,Jy,betx,bety,k1,k3):
    
    # Quadrupole contribution
    DQx =  1/(4*np.pi)*k1*betx
    DQy = -1/(4*np.pi)*k1*bety
        
    # Octupole contribution
    DQx += 3/(8*np.pi)*(k3/np.math.factorial(3))*(betx**2 * Jx - 2*betx*bety*Jy)
    DQy += 3/(8*np.pi)*(k3/np.math.factorial(3))*(bety**2 * Jy - 2*bety*betx*Jx)
    
    return DQx,DQy

    
def HeadOn_round_generating(t,Jx,Jy,emitt):
    term1 = 1/(1+t**2)*np.exp(-(Jx+Jy)/(2*emitt*(1+t)))
    term2 = sciSpec.iv(0,Jy/(2*emitt*(1+t)))
    term3 = sciSpec.iv(0,Jx/(2*emitt*(1+t)))-sciSpec.iv(1,Jx/(2*emitt*(1+t)))
    return term1*term2*term3
    
    
def HeadOn_round(Jx,Jy,emitt,xi):
        
    # Tune shifts, t is the integration variable
    DQx_n = integrate.quad(lambda t: HeadOn_round_generating(t,Jx,Jy,emitt), 0, np.inf)[0]
    DQy_n = integrate.quad(lambda t: HeadOn_round_generating(t,Jy,Jx,emitt), 0, np.inf)[0]
      
    return xi*DQx_n,xi*DQy_n

#================================================================================
#================================================================================
    
    
