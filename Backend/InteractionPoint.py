import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sciSpec
import warnings


import Backend.Constants as cst

class InteractionPoint:
    def __init__(self,IP,b1,b2):
        self.__type__ = 'IP'
        self.name = IP
        self.num  = IP[-1]
        
        self.b1 = copy.deepcopy(b1.at_IP(IP))
        self.b2 = copy.deepcopy(b2.at_IP(IP))
            
        # Dataframe with all beam-beam informations
        #============================================
        self.bb = self.make_bb_df()
        self.lr = self.bb.loc[self.bb.index.str.contains('bb_lr')]
        self.ho = self.bb.loc[self.bb.index.str.contains('bb_ho')]
              
        
    def make_bb_df(self,):
        
        # Using .values to drop the index since different for b1,b2
        _BB = pd.DataFrame({ 's'          : self.b1.bb['s_lab'],
                            ('b1','x_lab'): self.b1.bb['x_lab'].values,
                            ('b1','y_lab'): self.b1.bb['y_lab'].values,
                            ('b2','x_lab'): self.b2.bb['x_lab'].values,
                            ('b2','y_lab'): self.b2.bb['y_lab'].values,         
                            })
        # Note: these .get_ methods are interpolated, but completely identical to the value we would get at the 
        _BB.insert(1,'r'   ,self.b2.get_sigy(_BB['s'])/self.b2.get_sigx(_BB['s']))
        _BB.insert(2,'dx'  ,_BB[('b2','x_lab')]-_BB[('b1','x_lab')])
        _BB.insert(3,'dy'  ,_BB[('b2','y_lab')]-_BB[('b1','y_lab')])
        _BB.insert(4,'dx_n',_BB['dx']/self.b2.get_sigx(_BB['s']))
        _BB.insert(5,'dy_n',_BB['dy']/self.b2.get_sigy(_BB['s']))
        
        # Non-symmetric parameters
        _BB.insert(6,'A_w_s',self.b1.get_sigx(_BB['s'])/self.b2.get_sigy(_BB['s']))
        _BB.insert(7,'B_w_s',self.b1.get_sigy(_BB['s'])/self.b2.get_sigx(_BB['s']))
        
        # Weak beam beta function
        _BB.insert(8,'betx',self.b1.get_betx(_BB['s']))
        _BB.insert(9,'bety',self.b1.get_bety(_BB['s']))
        
        # Strong beam quadrupolar and octupolar component:
        _BB.insert(10,'k1',[self.b2.strong_knl(_dx,_dy)[0][1] for _dx,_dy in zip(_BB['dx'],_BB['dy'])])
        _BB.insert(11,'k3',[self.b2.strong_knl(_dx,_dy)[0][3] for _dx,_dy in zip(_BB['dx'],_BB['dy'])])
        
        # Making sure that the s location for both beams is compatible
        assert(np.all(np.array(_BB.s) == np.array(self.b2.bb.s_lab)))
        
        # Making sure that the interpolation at the marker location gives precisely the same value
        assert(np.all(np.array(_BB[('b2','x_lab')]) == self.b2.get_x_lab(self.b2.bb.s_lab)))
        
        return _BB
    
    def get_dx_dy(self,s):
        dx = self.b2.get_x_lab(s) - self.b1.get_x_lab(s)
        dy = self.b2.get_y_lab(s) - self.b1.get_y_lab(s)
        return dx,dy
    
    def get_dx_n_dy_n(self,s):
        dx,dy = self.get_dx_dy(s)
        dx_n  = dx/self.b2.get_sigx(s)
        dy_n  = dy/self.b2.get_sigy(s)
        return dx_n,dy_n
    

    
    
        
class Beam:
    def __init__(self,beam,twiss,survey,Nb=None,E=None,emittx_n=None,emitty_n=None,dp_p0 = None):
        self.__type__ = 'Beam'
        self.name = beam
        self.num  = beam[-1]
        self.twiss_full,self.survey_full  = twiss,survey
        self.twiss,self.survey  = twiss,survey
        self.bb,self.lr,self.ho = None,None,None
    
        # Beam properties
        self.Nb       = Nb
        self.E        = E
        self.emittx_n = emittx_n
        self.emitty_n = emitty_n
        self.dp_p0    = dp_p0
        
    # To extract around an IP
    #---------------------------
    def at_IP(self,IP):
        self.twiss,self.survey = extract_IP_ROI(IP,self.name,self.twiss_full,self.survey_full)
        
        #if self.name == 'b2':
        #    self.twiss['s']     *= -1
        #    self.twiss['s_lab'] *= -1
        
        # Shortcut for long range and head on markers
        self.bb = self.twiss.loc[self.twiss.index.str.contains('bb_')]
        self.lr = self.twiss.loc[self.twiss.index.str.contains('bb_lr')]
        self.ho = self.twiss.loc[self.twiss.index.str.contains('bb_ho')]
        
        return self
    
    # To find multipole expansion
    #---------------------------
    def strong_knl(self,dx,dy):
        IL_eq = -self.Nb*cst.elec*cst.c
        
        n = np.arange(12+1)
        integratedComp = -cst.mu0*(IL_eq)*sciSpec.factorial(n)/(2*np.pi)/(dx+1j*dy)**(n+1)
        _kn,_sn = np.real(integratedComp),np.imag(integratedComp)
        
        knl,snl = _kn/self.p0,_sn/self.p0
        return  knl,snl
        
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
        return self.Nb*cst.r_p*self.get_betx(s)/(2*np.pi*self.gamma_r*self.get_sigx(s)*(self.get_sigx(s)+self.get_sigy(s)))
    def get_xi_y(self,s):
        return self.Nb*cst.r_p*self.get_bety(s)/(2*np.pi*self.gamma_r*self.get_sigy(s)*(self.get_sigx(s)+self.get_sigy(s)))
    #-----


def extract_IP_ROI(IP,beam,twiss,survey):
    
    # ROI from dipoles
    #try:
    ROI_twiss  =  twiss.loc[f'mb.a8l{IP[-1]}.{beam}_dex':f'mb.a8r{IP[-1]}.{beam}_den'].copy()
    ROI_survey = survey.loc[f'mb.a8l{IP[-1]}.{beam}_dex':f'mb.a8r{IP[-1]}.{beam}_den'].copy()
    #except:
    #    ROI_twiss  =  twiss.loc[f'mb.a8l{IP[-1]}.{beam}':f'mb.a8r{IP[-1]}.{beam}'].copy()
    #    ROI_survey = survey.loc[f'mb.a8l{IP[-1]}.{beam}':f'mb.a8r{IP[-1]}.{beam}'].copy()
        
    # Angle for rotation of survey
    angle = -ROI_survey.loc[IP,'theta']
    
    # Re-centering before rotating
    z,x =  ROI_survey['z']-ROI_survey.loc[IP,'z'], ROI_survey['x']-ROI_survey.loc[IP,'x']
    zz = z*np.cos(angle) - x*np.sin(angle)
    xx = z*np.sin(angle) + x*np.cos(angle)
    
    # Inserting in dataframe
    ROI_survey.insert(1,'x_rot',xx)
    ROI_survey.insert(2,'y_rot',ROI_survey['y'])
    ROI_survey.insert(3,'z_rot',zz)
    ROI_survey.insert(4,'s_rot',ROI_survey['s']-ROI_survey.loc[IP,'s'])
    
    # Lab frame coordinates
    ROI_twiss.insert(1,'x_lab',ROI_twiss['x'] + ROI_survey['x_rot'])
    ROI_twiss.insert(2,'y_lab',ROI_twiss['y'] + ROI_survey['y_rot'])
    ROI_twiss.insert(3,'s_lab',ROI_twiss['s'] - ROI_twiss.loc[IP,'s'])
    
    
    return ROI_twiss,ROI_survey


