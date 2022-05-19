#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:50:29 2019
i
@author: pbelange
"""

import numpy as np

#============================================
#   PHYSICS CONSTANTS
#============================================
c = 299792458                                   # Speed of light [m/s]
mu0 = 4*np.pi*1e-7                              # Vacuum permeability [T*m/A]
eps0 = 1/(mu0*c**2)                             # Vacuum permitivity [F/m]
g = 9.81	                              		# Standard gravity [m/s^2]
elec = 1.602176634e-19                          # Elementary charge [C]
a0 = 52.917721067e-12                           # Bohr's radius [m]
kB = 1.380649e-23                               # Boltzman constant [J/K]
hP = 6.62607015e-34                             # Planck's constant [m^2*kg/s]
Na = 6.02214076e23                              # Avogadro's number
Mu = 1e-3                                       # Molar mass constant [kg/mol]

m_e =  9.10938356e-31                           # Electron mass [kg]
m_e_eV = (m_e*c**2)/elec                        # Electron mass [eV]
r_e = elec**2/(4*np.pi*eps0*m_e*c**2)           # Classical electron radius [m]
r_p = elec**2/(4*np.pi*eps0*m_p*c**2)           # Classical proton radius [m]
m_p = 1.672621898e-27                           # Proton mass [kg]
m_p_eV = (m_p*c**2)/elec                        # Proton mass [eV]



#============================================
#   LHC CONSTANTS
#============================================
LHC_C = 26659                                       # Circumference [m]
LHC_L_ARC_CELL = 53.45								# Length of a half arc cell [m]
LHC_RHO_ARC = 2804                                  # Bending radius of arc sections
LHC_F_REV = c/LHC_C                                 # Revolution frequency [Hz]
LHC_TURN_LENGTH = LHC_C/c                           # Turn duration [s]
LHC_RF_INJ = 400.787e6                              # RF frequency at injection [Hz]
LHC_RF_TOP = 400.788e6                              # RF frequency at top energy [Hz]
LHC_N_BUCKETS = 35640                               # Number of RF buckets
LHC_BUCKET_LENGTH = LHC_TURN_LENGTH/LHC_N_BUCKETS   # Bucket duration [s]
LHC_H_BEAM_SCREEN = 36.9e-3                         # Height of the beam screen [m]
LHC_W_BEAM_SCREEN = 46.5e-3                         # Width of the beam screen [m]
LHC_ALPHA_BEAM_SCREEN = 52.4                        # Angle of the end of the flat part [deg]
LHC_BETA_BEAM_SCREEN = 37.6                        	# Complimentary to alpha [deg]

# Cross sectional area of the beam screen:
LHC_AREA_BEAM_SCREEN = np.pi*(LHC_W_BEAM_SCREEN/2)**2*(1-4*LHC_BETA_BEAM_SCREEN/360) + LHC_W_BEAM_SCREEN*np.sin(np.deg2rad(LHC_BETA_BEAM_SCREEN))*LHC_H_BEAM_SCREEN/2

# Perimeter of the cross section of the beam screen
LHC_PERIMETER_BEAM_SCREEN = (LHC_W_BEAM_SCREEN/2)*4*(np.deg2rad(LHC_ALPHA_BEAM_SCREEN)+np.sin(LHC_BETA_BEAM_SCREEN))
