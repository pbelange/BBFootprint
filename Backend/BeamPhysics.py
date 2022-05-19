import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
def normCoordinates(x,px,alpha=None,beta=None,SVD=False):
    
    if SVD:
        alpha,beta = computeAlphaBeta(x,px)
        
    #N0 = [[1/np.sqrt(beta),0],[alpha/np.sqrt(beta), np.sqrt(beta)]]
    x_n  = x/np.sqrt(beta)
    px_n = alpha*x/np.sqrt(beta) + px*np.sqrt(beta)
    
    return x_n,px_n
##############################################################


##############################################################
def computeAlphaBeta(x,px):
    '''Taken from https://arxiv.org/pdf/2006.10661.pdf '''
    
    U,s,V= np.linalg.svd([x,px])         #SVD
    
    N = np.dot(U,np.diag(s))
    theta = np.arctan(-N[0,1]/N[0,0])    #AngleofR(theta)
    co=np.cos(theta) ; si=np.sin(theta)
    
    R = [[co,si],[-si,co]]   
    X = np.dot(N,R)                      #Floquetupto1/det(USR)
    
    beta = np.abs(X[0,0]/X[1,1])
    alpha = X[1,0]/X[1,1]
    
    # dropped
    ex =s[0]*s[1]/(len(x)/2.)            #emit=det(S)/(n/2)
    
    return alpha,beta

##############################################################



##############################################################
def getAction(x,px,alpha=None,beta=None,SVD=False):
    
    if SVD:
        alpha,beta = computeAlphaBeta(x,px)
    gamma = (1+alpha**2)/beta
    
    J = (gamma*x**2  + 2*alpha*x*px + beta*px**2)/2
    
    return J
##############################################################



#############################################################

def generateCoord(Jx,alpha,beta,NParticles,plane='x'):
    
    gamma = (1+alpha**2)/beta
    
    phi = np.linspace(0,2*np.pi,NParticles+1)[1:]
    x  = np.sqrt(2*beta*Jx)*np.cos(phi)
    px = -np.sqrt(2*Jx/beta)*(np.sin(phi)+alpha*np.cos(phi))
    
    coordinates = pd.DataFrame({f'J{plane}':Jx*np.ones(len(x)),f'{plane}':x,f'p{plane}':px})
    
    return coordinates
    

#############################################################


#############################################################
def plotWorkingDiagram(order = 12,QxRange=np.array([0,1]),QyRange=np.array([0,1]),**kwargs):

    # Initialization
    options = {'color':'k','alpha':0.5}
    options.update(kwargs)
    QxRange,QyRange = np.array(QxRange),np.array(QyRange)
    def intList(n): return np.arange(-n,n+1)
    plt.axis('square')
    plt.xlim(QxRange)
    plt.ylim(QyRange)


    # Creating all combinations except vertical lines
    popt = []
    for m1,m2,n in itertools.product(intList(order),intList(order)[intList(order)!=0],intList(200)):
        if np.abs(m1)+np.abs(m2) <= order:
            popt.append((-m1/m2,n/m2))


    # Removing Duplicates
    # TODO: change this line in order to keep track of the order of the resonance
    popt = list(set(popt))

    # Keeping only lines in ROI
    ROI_popt = []
    for slope,y0 in popt:
        line = slope*QxRange + y0

        if np.any(np.logical_and(line>=np.min(QyRange),line<=np.max(QyRange))):
            ROI_popt.append((slope,y0))

    # Plotting
    regularSlopes = np.array(ROI_popt)[:,0]
    for slope,y0 in ROI_popt:
        plt.plot(QxRange,slope*QxRange + y0,**options)

        # Reflection around y=x to take care of the cases where m2=0
        with np.errstate(divide='ignore'):
            invertedSlope = (np.diff(QyRange)/np.diff(slope*QyRange + y0))[0]
        if not np.round(invertedSlope,5) in list(np.round(regularSlopes,5)):
            plt.plot(slope*QyRange + y0,QyRange,**options)

#############################################################
