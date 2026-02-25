#!/usr/bin/env python3
"""
Rescale a stepsic h-independent (Mpc) HDF5 IC into the standard Gadget4 Mpc/h convention.

Gadget4 enforces Hubble=100 (the Mpc/h convention) and TERMINATEs otherwise.  stepsic
ICs generated with HINDEPENDENT=false are in physical Mpc with H0 = 100h km/s/Mpc, so
they must be converted before Gadget4 will accept them.  The physics is unchanged; only
the h-bookkeeping is:

    coordinates [Mpc]   ->  [Mpc/h]      : x   *= h
    BoxSize     [Mpc]   ->  [Mpc/h]      : L   *= h
    masses  [1e11 Msun] ->  [1e10 Msun/h]: m   *= 10*h     (== 1e11*h/1e10)
    velocities [km/s]                    : unchanged (physical peculiar / sqrt(a))

Then run Gadget4 with: Hubble=100, HubbleParam=h, UnitLength_in_cm=3.085678e24,
UnitMass_in_g=1.989e43, UnitVelocity_in_cm_per_s=1e5, and BoxSize/softening in Mpc/h.

Usage:
    python rescale_ic_to_gadget.py IN.hdf5 OUT.hdf5 [--mass-unit-1e11]
"""
import argparse, shutil, h5py, numpy as np

def rescale(src, dst, mass_in_1e11=True):
    shutil.copy(src, dst)
    with h5py.File(dst, 'r+') as h:
        H = h['Header'].attrs
        hub = float(H['HubbleParam'])
        p = h['PartType1']
        p['Coordinates'][...] = p['Coordinates'][:] * hub
        # stepsic HINDEPENDENT masses are in 1e11 Msun; Gadget wants 1e10 Msun/h
        mfac = (10.0 * hub) if mass_in_1e11 else hub
        if 'Masses' in p:
            p['Masses'][...] = p['Masses'][:] * mfac
        L_old = float(H['BoxSize']); H['BoxSize'] = L_old * hub
        print(f"  h = {hub}")
        print(f"  BoxSize: {L_old:.3f} Mpc -> {float(H['BoxSize']):.3f} Mpc/h")
        if 'Masses' in p:
            print(f"  Mass[0]: -> {float(p['Masses'][0]):.4f}  (1e10 Msun/h)")
        print(f"  coord max (Mpc/h): {p['Coordinates'][:].max(0).round(2)}")
    print(f"  wrote {dst}")
    print(f"  Gadget4 param: Hubble=100  HubbleParam={hub}  BoxSize={L_old*hub:.3f}  "
          f"UnitMass_in_g=1.989e43")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst')
    ap.add_argument('--mass-unit-1e10', action='store_true',
                    help='IC masses are already 1e10 Msun (not the stepsic 1e11 default)')
    a = ap.parse_args()
    rescale(a.src, a.dst, mass_in_1e11=not a.mass_unit_1e10)
