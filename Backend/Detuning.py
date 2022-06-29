#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022
@author: pbelange
Description :
"""

import numpy as np
import scipy.integrate as integrate
import scipy.special as sciSpec

import Backend.Constants as cst



#================================================================================
#    Full analytic detuning
#================================================================================

def DQx_DQy(ax,ay,r,dx_n,dy_n,xi,A_w_s,B_w_s,method='fast'):
    """
    Notes: 
    The function expects an array for ax,ay, and a single value for the other parameters
    --------
    ax,ay -> normalized amplitude, ax = x/sigma_weak
    r     -> sigma_y/sigma_x
    dx,sy -> normalized bb separation, dx_n = dx/sigma_strong
    xi    -> beam-beam parameter
    A_w_s -> sigma_w_x/sigma_s_y
    B_w_s -> sigma_w_y/sigma_s_x
    """
    DQx = np.array([2*xi*dC00dx(A_w_s*_ax,B_w_s*_ay,r,dx_n,dy_n,method)/(A_w_s*_ax) for _ax,_ay in zip(ax,ay)])
    DQy = np.array([2*xi*dC00dy(A_w_s*_ax,B_w_s*_ay,r,dx_n,dy_n,method)/(B_w_s*_ay) for _ax,_ay in zip(ax,ay)])
    return (A_w_s**2)*DQx,(B_w_s**2)*DQy

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


def Bess2D_INT(X,Y,n):

    # Choosing between the real part or the imaginary part depending on n
    generatingFun,sign = {0:(Bess2D_generating_real,(-1)**(n/2)),
                          1:(Bess2D_generating_imag,(-1)**((n-1)/2))}[n%2]
    # computing the integral
    integratedFactor = integrate.quad(lambda phi: generatingFun(phi,X,Y,n), 0, 2*np.pi)[0]

    return sign*np.exp(-X-2*Y)/2/np.pi * integratedFactor


def Bess2D_SUM(X,Y,n):
    qmax = 40
    order = np.arange(-qmax,qmax+1)
    
    q,XX = np.meshgrid(order,X)
    _,YY = np.meshgrid(order,Y)
    
    return np.exp(-X-Y)*np.sum(sciSpec.iv(n-2*q,XX)*sciSpec.iv(q,YY),axis=1)


def Bess2D(X,Y,n,method='int'):
    return {'int':Bess2D_INT,'sum':Bess2D_SUM}[method](X,Y,n)





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

def dQ0daz_Bess_generating(phi,X,Y,azbar,dzbar):
    arg =  -X*np.sin(phi) + 2*Y*np.sin(phi)**2
    # exp(arg) is commong to Bess_0, Bess_1, Bess_2. The prefactor coming from the sum is:
    pre = azbar*(np.cos(2*phi)-1)/2 - dzbar*np.sin(phi)
    return pre*np.exp(arg)

def Fast_dQ0daz(t,azbar,dzbar,etaz):
    X    =  t*azbar*dzbar
    Y    = -t*azbar**2/4
    
    integratedFactor = integrate.quad(lambda phi: dQ0daz_Bess_generating(phi,X,Y,azbar,dzbar), 0, 2*np.pi)[0]
    
    return etaz*np.exp(-t/2*(azbar-dzbar)**2)*(np.exp(-X-2*Y)/2/np.pi)*integratedFactor
#---------------------------------------


#---------------------------------------
def dC00dx_generating(t,ax,ay,r,dx_n,dy_n,method='regular'):
    # bar variables
    axbar,aybar,dxbar,dybar = ax*r, ay/g(t,r) , dx_n , r*dy_n/g(t,r)

    # modified version of eta to include all the prefactors
    etax_modif = r/g(t,r)
    
    derivative = {'regular':dQ0daz,'fast':Fast_dQ0daz}[method]

    return derivative(t,axbar,dxbar,etax_modif)*Q0z(t,aybar,dybar)

def dC00dy_generating(t,ax,ay,r,dx_n,dy_n,method='regular'):
    # bar variables
    axbar,aybar,dxbar,dybar = ax*r, ay/g(t,r) , dx_n , r*dy_n/g(t,r)

    # modified version of eta to include all the prefactors
    etay_modif = 1/(g(t,r)**2)
    
    derivative = {'regular':dQ0daz,'fast':Fast_dQ0daz}[method]

    return derivative(t,aybar,dybar,etay_modif)*Q0z(t,axbar,dxbar)


def dC00dx(ax,ay,r,dx_n,dy_n,method = 'regular'):
    return integrate.quad(lambda t: dC00dx_generating(t,ax,ay,r,dx_n,dy_n,method), 0, 1)[0]


def dC00dy(ax,ay,r,dx_n,dy_n,method = 'regular'):
    return integrate.quad(lambda t: dC00dy_generating(t,ax,ay,r,dx_n,dy_n,method), 0, 1)[0]
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








def Z1(x):
    return np.exp(-x)*(sciSpec.iv(0,x)-sciSpec.iv(1,x))

def Z2(x):
    return np.exp(-x)*sciSpec.iv(0,x)

def HeadOn_round_generating(t,alphax,alphay,r):
    # Assume in x direction. For y, change x->y, y->x, r->1/r
    # See: https://inspirehep.net/files/d7fed02b4b59558edf043a65f0f92049
    prefactor = 1/((1+t)**(3/2)* (1+t/r**2)**(1/2))
    return (1+1/r)/2* prefactor * Z1(alphax/(1+t)) * Z2(alphay/(1+t/r**2))
    
    
def HeadOn_round(ax,ay,r,emitt,xi):
    
    if isinstance(emitt, (int, float)):
        emitt = emitt*np.ones(2)
    if isinstance(xi, (int, float)):
        xi = xi*np.ones(2)
        
    # Normalization from Chao
    alphax = ax**2/(2*(1+emitt[1]/emitt[0]))
    alphay = ay**2/(2*(1+emitt[0]/emitt[1]))
    
    # Tune shifts, t is the integration variable
    DQx_n = np.array([integrate.quad(lambda t: HeadOn_round_generating(t,_alphax,_alphay,r)  , 0, np.inf)[0] for _alphax,_alphay in zip(alphax,alphay)])
    DQy_n = np.array([integrate.quad(lambda t: HeadOn_round_generating(t,_alphay,_alphax,1/r), 0, np.inf)[0] for _alphax,_alphay in zip(alphax,alphay)])
      
    return -xi[0]*DQx_n,-xi[1]*DQy_n




# OLD

    
#def HeadOn_round_generating(t,Jx,Jy,emitt):
#    term1 = 1/(1+t**2)*np.exp(-(Jx+Jy)/(2*emitt*(1+t)))
#    term2 = sciSpec.iv(0,Jy/(2*emitt*(1+t)))
#    term3 = sciSpec.iv(0,Jx/(2*emitt*(1+t)))-sciSpec.iv(1,Jx/(2*emitt*(1+t)))
#    return term1*term2*term3
#    
#    
#def HeadOn_round(Jx,Jy,emitt,xi):
#        
#    # Tune shifts, t is the integration variable
#    DQx_n = np.array([integrate.quad(lambda t: HeadOn_round_generating(t,_Jx,_Jy,emitt), 0, np.inf)[0] #for _Jx,_Jy in zip(Jx,Jy)])
#    DQy_n = np.array([integrate.quad(lambda t: HeadOn_round_generating(t,_Jy,_Jx,emitt), 0, np.inf)[0] #for _Jx,_Jy in zip(Jx,Jy)])
#      
#    return -xi*DQx_n,-xi*DQy_n


#================================================================================
#================================================================================
    
    
