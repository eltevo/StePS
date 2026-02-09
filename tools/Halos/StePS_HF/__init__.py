'''
StePS halo finder package.
'''
import numpy as np


_VERSION="v0.3.0.0"
_YEAR="2024-2026"
_AUTHOR="Gabor Racz"

# Global variables (constants)
G  = 4.30091727003628e-09 # gravitational constant G in Mpc/Msol*(km/s)^2 units (In Rockstar: 4.30117902e-9)
H2RHO_C = 3.0/(G*8.0*np.pi)*1.0e4 / 1.0e11 # Comoving critical density in 1e11 h^2*Msol/Mpc^3 units (here: 2.7753662724583 * 1e11 h^2*Mpc/Msol^3; Rockstar: 2.77519737e11 h^2*Mpc/Msol^3)
# usual StePS internal units
UNIT_T=47.148299511187325 #Unit time in Gy
UNIT_V=20.738652969844207 #Unit velocity in km/s
UNIT_D=3.085677581491367e+24 #=1Mpc Unit distance in cm