
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


import Backend.Constants as cst

class InteractionPoint:
    def __init__(self,IP,b1,b2):
        self.__type__ = 'IP'
        self.name = IP
        self.num  = IP[-1]
        
        self.b1 = b1
        self.b2 = b2
    
        # Forcing the position of BBLR interactions
        _tmp = np.arange(0,150,7.5/2)[1:]
        self.s_BBLR = np.concatenate((-np.flip(_tmp),_tmp))
        
        # Computing the separation between the beams in each plane
        self.xsep,self.ysep = self.get_xsep_ysep(self,self.s_BBLR)
    
    def get_xsep_ysep(self,s):
        Dx = self.b2.get_x_lab(s) - self.b1.get_x_lab(s)
        Dy = self.b2.get_y_lab(s) - self.b1.get_y_lab(s)
        return Dx,Dy
        
class Beam:
    def __init__(self,beam,Nb=None,E=None,emittx_n=None,emitty_n=None,dp_p0 = None)
        self.__type__ = 'Beam'
        self.name = beam
        self.num  = beam[-1]
        self.twiss,self.survey = None,None
    
        # Beam properties
        self.Nb       = Nb
        self.E        = E
        self.emittx_n = emittx_n
        self.emitty_n = emitty_n
        self.dp_p0    = dp_p
        
    # To extract around an IP
    #---------------------------
    def at_IP(self,IP,twiss,survey):
        self.twiss,self.survey = extract_IP_ROI(IP,self.name,twiss,survey)
        return self


    # Some additionnal properties
    #---------------------------
    @property
    def p0(self):
        return np.sqrt(self.E**2-cst.m_p_eV**2)/cst.c
    
    @property
    def gamma_r(self):
        return 1+self.E/cst.m_p_eV
    
    @property
    def beta_r(self):
        return np.sqrt(1-1/self.gamma_r**2)
    
    @property
    def emittx(self):
        return self.emittx_n/self.gamma_r
    
    @property
    def emitty(self):
        return self.emitty_n/self.gamma_r
    
    @property
    def xi(self):
        if self.emittx_n != self.emitty_n:
            warnings.warn("Beam.xi is only defined for round beams, taking the average... (use Beam.get_xi_x,Beam.get_xi_y instead)")
            round_emitt = np.mean([self.emittx_n,self.emitty_n])
        else:
            round_emitt = self.emittx_n
        return self.Nb*cst.r_p/(4*np.pi*round_emitt)
    #---------------------------
    

    # Beam optics
    #---------------------------
    #-----
    def get_x_lab(self,s):
        return np.interp(s, self.twiss['s_lab'],self.twiss['x_lab'])
    def get_y_lab(self,s):
        return np.interp(s, self.twiss['s_lab'],self.twiss['y_lab'])
    #-----
    def get_betx(self,s):
        return np.interp(s, self.twiss['s_lab'],self.twiss['betx'])
    def get_bety(self,s):
        return np.interp(s, self.twiss['s_lab'],self.twiss['bety'])
    #-----
    def get_dpx(self,s):
        return np.interp(s, self.twiss['s_lab'],self.twiss['dpx'])
    def get_dpy(self,s):
        return np.interp(s, self.twiss['s_lab'],self.twiss['dpy'])
    #-----
    def get_sigx(self,s):
        return np.sqrt(self.get_betx(s)*self.emittx + (self.get_dpx(s)*self.dp_p0)**2)
    def get_sigy(self,s):
        return np.sqrt(self.get_bety(s)*self.emitty)
    #-----
    def get_xi_x(self,s):
        return -self.Nb*cst.r_p*self.get_betx(s)/(2*np.pi*self.gamma_r*self.get_sigx(s)*(self.get_sigx(s)+self.get_sigy(s)))
    def get_xi_y(self,s):
        return -self.Nb*cst.r_p*self.get_bety(s)/(2*np.pi*self.gamma_r*self.get_sigy(s)*(self.get_sigx(s)+self.get_sigy(s)))
    #-----

def extract_IP_ROI(IP,beam,twiss,survey):
    
    # ROI from dipoles
    ROI_twiss  =  twiss.loc[f'mb.a8l{IP[-1]}.{beam}_dex':f'mb.a8r{IP[-1]}.{beam}_den'].copy()
    ROI_survey = survey.loc[f'mb.a8l{IP[-1]}.{beam}_dex':f'mb.a8r{IP[-1]}.{beam}_den'].copy()

    # Angle for rotation of survey
    angle = ROI_survey.loc[IP,'theta']+3*np.pi/2
    
    x,z =  ROI_survey['x'], ROI_survey['z']
    xx = x*np.cos(angle) - z*np.sin(angle)
    zz = x*np.sin(angle) + z*np.cos(angle)
    
    # Inserting in dataframe
    ROI_survey.insert(1,'x_rot',xx)
    ROI_survey.insert(2,'y_rot',ROI_survey['y'])
    ROI_survey.insert(3,'z_rot',zz)
    ROI_survey.insert(4,'s_rot',ROI_survey['s']-ROI_survey.loc[IP,'s'])
    
    # Lab frame coordinates
    ROI_twiss.insert(1,'x_lab',ROI_twiss['x'] + ROI_survey['z_rot'])
    ROI_twiss.insert(2,'y_lab',ROI_twiss['y'] + ROI_survey['y_rot'])
    ROI_twiss.insert(3,'s_lab',ROI_twiss['s'] - ROI_twiss.loc[IP,'s'])
    
    
    return ROI_twiss,ROI_survey
