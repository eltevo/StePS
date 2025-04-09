#!/usr/bin/env python3

import numpy as np
import astropy.units as u
from astropy.units import cds
cds.enable()

_VERSION = "1.0.0"
_YEAR = "2018-2025"
_AUTHOR = "Gabor Racz"

# This script calculates the internal units of the StePS code.
# Welcome message
print("Welcome to the StePS code unit calculator version", _VERSION)
print("\tCopyright (C) %s by %s" % (_YEAR, _AUTHOR))
print("\tThis program comes with ABSOLUTELY NO WARRANTY.")
print("\tThis is free software, and you are welcome to redistribute it")
print("\tunder certain conditions; see the file LICENSE for details.")

print("\nThis simple script calculates the internal units of the StePS code and prints them.")
print("The calculation is based on the astropy units package.")


g= 1.0*cds.G
print("\nGravitational constant in SI:")
print(g.to(u.m * u.m * u.m / u.kg /u.s /u.s))
Mpc = 1.0*u.Mpc
print("\nMpc in SI:")
print(Mpc.to(u.m))
M_unit = 1.0e11*u.solMass
print("\n10e11M_sol in SI:")
print(M_unit.to(u.kg))

print("[T] = sqrt([D]^3/[M]/[G])")
print("\nWe set G=1, [D]=Mpc, [M]=10e11M_sol, so the time unit will be")
T = np.sqrt(Mpc.to(u.m).value**3/M_unit.to(u.kg).value/g.to(u.m * u.m * u.m / u.kg /u.s /u.s).value) * u.s
print("[T] = ",T)
print("\n\nThe internal StePS units are:\n-----------------------------")
print("Time:\t\t[T]  = ", T, "  = ", T.to(u.Gyr))
V=Mpc/T
print("Velocity:\t[V]  = ", V.to(u.km/u.s))
print("Distance:\t[D]  = ", Mpc.to(u.cm), " =", Mpc)
print("Mass\t\t[M]  = ", M_unit.to(u.kg), " = %e solMass" % M_unit.to(u.solMass).value)

print("\n\nThe internal StePS_HF constants are:\n------------------------------------")
print("Gravitational constant (in Mpc/Msol*(km/s)^2 units):\tG     =", g.to(u.Mpc / u.solMass * (u.km / u.s)**2))
rho_crit = 3.0 /( g.to(u.Mpc / u.solMass * (u.km / u.s)**2).value * 8.0 * np.pi)*100.0**2
print("Critical_density at z=0 (in h^2*Msol/Mpc^3 units):\trho_c = %.14e (solMass h2) / Mpc3"% rho_crit)
print("\n")