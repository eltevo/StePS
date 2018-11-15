#!/usr/bin/env python3

import numpy as np
import h5py
import sys
import time

#Beginning of the script
if len(sys.argv) != 3:
    print("Error:")
    print("usage: ./hdf52ascii_snapshot.py <input HDF5 snapshot> <output ASCII snapshot>\nExiting.")
    sys.exit(2)

#reading and converting the HDF5 file
print("Reading the input HDF5 file...")
start = time.time()
HDF5_snapshot = h5py.File(str(sys.argv[1]), "r")
N = np.int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1])
M_min = HDF5_snapshot['/Header'].attrs['MassTable'][1]
R_max = HDF5_snapshot['/Header'].attrs['Lbox']
print("Number of particles:\t%i\nMinimal mass:\t%f*10e11M_sol\nLinear size:\t%fMpc" % (N,M_min,R_max))
ASCII_snapshot = np.zeros((N,7), dtype=np.float64)
ASCII_snapshot[:,0:3] = HDF5_snapshot['/PartType1/Coordinates']
ASCII_snapshot[:,3:6] = HDF5_snapshot['/PartType1/Velocities']
ASCII_snapshot[:,6] = HDF5_snapshot['/PartType1/Masses']
HDF5_snapshot.close()
end = time.time()
print("..done in %fs. \n\n" % (end-start))
#Saving the ascii file
outputfilename = str(sys.argv[2])
print("Saving the %s ASCII file..." % outputfilename)
start = time.time()
np.savetxt(outputfilename, ASCII_snapshot, delimiter='\t')
end = time.time()
print("..done in %fs. \n\n" % (end-start))
