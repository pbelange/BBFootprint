import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import matplotlib.transforms as transforms
import scipy.special as sciSpec
import subprocess

import Backend.Constants as cst





##############################################################
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)
##############################################################


##############################################################
def vecNorm(vec):
    return np.sqrt(vec[0]**2 + vec[1]**2)
##############################################################


##############################################################
def getMultiCoeff(I,L,pos,nmax=1):   
    # compute the integrated normal and skew components
    # Note: mistake in Axel's thesis, the n! not consistent with definition of fields
    # Note: mistake in Axel's thesis, sign of integrated component not correct. Refer to S.F. paper.
    n = np.arange(nmax+1)
    integratedComp = -cst.mu0*(I*L)*sciSpec.factorial(n)/(2*np.pi)/(pos[0]+1j*pos[1])**(n+1)
    
    kn,sn = np.real(integratedComp),np.imag(integratedComp)
    
    return kn,sn
##############################################################

                               
                               
##############################################################
class wire():
    def __init__(self,x,y,I,L,madClass=None):
        self.x,self.y = x,y
        self.r0,self.theta0 = cart2pol(self.x,self.y)
        self.I = I
        self.L = L
        self.madClass = madClass
        if madClass is not None:
            self.toMad = {'multipole':self.toMad_multipole,
                          'bb':self.toMad_bb,
                          'tobias':self.toMad_tobias}[madClass.lower()]
        
    def getField(self,xObs,yObs,order=np.inf):
        self.order = order
        Bref = cst.mu0*self.I/(2*np.pi)
        
        if order != np.inf:
            complexFields = np.zeros(np.shape(xObs)).astype('complex128')
            for n in np.arange(1,order+1):
                complexFields += (Bref/self.r0)*(-np.exp(-1j*n*self.theta0))*(xObs/self.r0+1j*yObs/self.r0)**(n-1)
        else:
            pos_wire = self.x + 1j*self.y
            pos_Obs  = xObs   + 1j*yObs
            complexFields = Bref * (np.conjugate(pos_Obs) - np.conjugate(pos_wire)) / np.absolute(pos_Obs - pos_wire)**2
                
        
        # complexFields =  By + iBx
        # return Bx,By
        return np.imag(complexFields),np.real(complexFields)
    
    def getMultipole_strengths(self,order=20,normalise_at_E = None):
        # Computing integrated coefficients
        self.order = order
        kn,sn = getMultiCoeff(self.I,self.L,[self.x,self.y],nmax=order)
        
        # p0 -> momentum in eV/c
        if normalise_at_E is not None:
            p0 = np.sqrt(normalise_at_E**2-cst.m_p_eV**2)/cst.c
            kn = (kn/p0)
            sn = (sn/p0)

        # return kn,sn
        return kn,sn
    
    
    
    def getKick(self,xObs,yObs,Energy,order=np.inf):
        self.order = order
        
        if order != np.inf:
            kn,sn = getMultiCoeff(self.I,self.L,[self.x,self.y],nmax=order)

            p0 = np.sqrt(Energy**2-cst.m_p_eV**2)/cst.c
            KNL = (kn/p0)
            KSL = (sn/p0)

            complexKicks = np.zeros(np.shape(xObs)).astype('complex128')
            for n,KNL_n,KSL_n in zip(np.arange(order+1),KNL,KSL):
                complexKicks += -(KNL_n + 1j*KSL_n)*(xObs+1j*yObs)**(n)/sciSpec.factorial(n)

            # complexKicks =  DPx - iDPy
            # return DPx,DPy
            return np.real(complexKicks),-np.imag(complexKicks)
        else:
            
            q =  1 # when p0c given in eV
            p0 = np.sqrt(Energy**2-cst.m_p_eV**2)/cst.c
            
            rVec = vecNorm([self.x-xObs,self.y-yObs])
            amplitude = q*self.L*(cst.mu0*self.I/(2*np.pi*rVec))/(p0)
            
            Px = amplitude*(self.x-xObs)/rVec
            Py = amplitude*(self.y-yObs)/rVec
            
            return Px,Py
            
    
    def plotWireLocation(self,scaling=1,ax = None):
        if ax is None:
            ax = plt.gca()
            
        ax.plot([scaling*self.x],[scaling*self.y],'o',color='C1',fillstyle = 'none',markersize=12)
        if self.I>0:
            ax.plot([scaling*self.x],[scaling*self.y],'.',color='C1',markersize=8)                  
        else:
            ax.plot([scaling*self.x],[scaling*self.y],'x',color='C1',markersize=8)


    def toMad_multipole(self,at,Energy,name='wire_multipole',order=20,BBORBIT = False):

        # Computing integrated coefficients
        self.order = order
        kn,sn = getMultiCoeff(self.I,self.L,[self.x,self.y],nmax=order)
        
        # p0 -> momentum in eV/c
        p0 = np.sqrt(Energy**2-cst.m_p_eV**2)/cst.c
        KNL = (kn/p0)
        KSL = (sn/p0)

        # computing HKICK and VKICK from dipole component:
        #-----------------------------------
        # complexKicks =  DPx - iDPy
        complexKick = -(KNL[0] + 1j*KSL[0])
        HKICK,VKICK = np.real(complexKick),-np.imag(complexKick)
        #-----------------------------------


        def_1of2 = f'class_{name}_1of2 : MULTIPOLE,KNL = {{0,{",".join(KNL[1:].astype(str))}}},KSL = {{0,{",".join(KSL[1:].astype(str))}}};'
        def_2of2 = f'class_{name}_2of2 : KICKER, L=0, HKICK={HKICK}, VKICK={VKICK}, TILT=real;'

        

        if BBORBIT:
            thisWire = pd.DataFrame({'mode':['install','install'],
                         'name':[f'{name}_1of2',f'{name}_2of2'],
                         'at': [at,at],
                         'definition':[def_1of2 ,def_2of2 ]}) 
        else:
            thisWire = pd.DataFrame({'mode':['install'],
                                 'name':[f'{name}'],
                                 'at': [at],
                                 'definition':[def_1of2]}) 

        return thisWire
    
    
    def toMad_bb(self,at,Energy,name='wire_bb'):
        
        # Computing number of protons in strong beam
        gamma_r = Energy/cst.m_p_eV + 1
        beta_r = np.sqrt(1-1/gamma_r**2)
        N_p = int(self.I*cst.LHC_C/(cst.elec*cst.c)/((1+beta_r**2)/beta_r))
        
        # Effective bb charge (counter-rotating beam -> twice bb interactions over a length L)
        charge = int(2*N_p/cst.LHC_C*self.L)
        
        # Installation dataframe
        thisWire = pd.DataFrame({'mode':['install'],
                                 'name':[f'{name}'],
                                 'at': [at],
                                 'definition':[f'class_{name} : beambeam, '\
                                                f'charge = {-charge},'\
                                                f'xma = {self.x}, yma = {self.y},'\
                                                'sigx = 1e-6, sigy = 1e-6,'\
                                                'width=1,'\
                                                'BBDIR = -1;']})
        return thisWire
    
    def toMad_tobias(self,at,Energy = '_',name='wire_tobias'):
        # TODO:
        # CHANGE x-y inversion in the backend definition of wire object! 
        
        # No need for energy parameter
        _ = Energy
        
        thisWire = pd.DataFrame({'mode':['install'],
                                 'name':[f'{name}'],
                                 'at': [at],
                                 'definition':[f'class_{name} : wire,'\
                                               f'current = {self.I},' \
                                               f'L = 0,'\
                                               f'L_phy = {self.L},'\
                                               f'L_int = {self.L},'\
                                               f'Xma = {-self.x},'\
                                               f'Yma = {-self.y};']})
        
        return thisWire

##############################################################
                               
                               
                               
##############################################################
def plotVecField(X,Y,vecX,vecY,rValid=np.infty,scaling = 1,color = [],colorLim = [],colorLabel = '',mode='streamplot',nSeeds = 0,seed_points = None,seedAngle = 0,arrowColor = 'k',arrowSize = 1,norm = None,quiverDensity = 1):
 
    
    currentExtent =  [scaling*np.min(X),scaling*np.max(X),scaling*np.min(Y),scaling*np.max(Y)]
    
    #invalidRegion = ((X)**2 + (Y)**2 > (rValid)**2)
    copyX = X.copy()
    copyY = Y.copy()
    #X[invalidRegion] = np.nan
    #Y[invalidRegion] = np.nan
    #vecX[invalidRegion] = np.nan
    #vecY[invalidRegion] = np.nan
    
    currentNorm = vecNorm([vecX,vecY])

    if len(color)==0:
        colorLim = [np.min(currentNorm), np.max(currentNorm)] if len(colorLim) == 0 else colorLim
        if norm == 'log':
            plt.imshow(currentNorm, extent = currentExtent,origin='lower',norm = LogNorm(vmin=colorLim[0], vmax=colorLim[1]))
        else:
            plt.imshow(currentNorm, extent = currentExtent,origin='lower',vmin=colorLim[0], vmax=colorLim[1])    
    else:
        colorLim = [np.min(color), np.max(color)] if len(colorLim) == 0 else colorLim
        plt.imshow(color, extent = currentExtent,origin='lower',norm=norm,vmin=colorLim[0], vmax=colorLim[1])
    plt.colorbar(label = colorLabel)

    if mode == 'quiver':
        density = int(10/quiverDensity)
        plt.quiver(scaling*X[::density,::density],scaling*Y[::density,::density], vecX[::density,::density]/currentNorm[::density,::density],vecY[::density,::density]/currentNorm[::density,::density])#,width = maxArrowSize *0.05*scaling*(np.max(X)-np.min(X)),color = arrowColor)
    elif mode == 'streamplot':
        if seed_points is None:
            rVec = np.linspace(-rValid,rValid,nSeeds)
            seed_points = np.array([rVec*np.cos(seedAngle),rVec*np.sin(seedAngle)])
            
        plt.streamplot(scaling*copyX,scaling*copyY,vecX,vecY,density = 20,linewidth=arrowSize,color = arrowColor,start_points=scaling*seed_points.T)
##############################################################
                               
                               
                               
                               
##############################################################
                            
def plotBeamDirection(x,y,name= 'Beam 1',color = 'b',direction = '+',fontsize = 15,ax = None):
    if ax is None:
        ax = plt.gca()

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    ax.text(x,y,'    '+name, 
            transform=ax.transAxes, 
            fontsize=fontsize,
            color=color,
            verticalalignment='center',
            horizontalalignment='left', 
            bbox=props)

    boxZorder = [TOject.zorder for TOject in ax.texts if name in TOject.get_text()][0]
    xOffset = 0.02
    ax.plot(x+xOffset,y,'o',transform = ax.transAxes,color=color,fillstyle = 'none',markersize=12,zorder = boxZorder+1)
    if direction == '+':
        ax.plot(x+xOffset,y,'.',transform = ax.transAxes,color=color,markersize=8,zorder = boxZorder+1)
    else:
        ax.plot(x+xOffset,y,'x',transform = ax.transAxes,color=color,markersize=8,zorder = boxZorder+1)                       

##############################################################