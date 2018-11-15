#!/usr/bin/env python3

import numpy as np
import h5py

def writeHDF5snapshot(dataarray, outputfilename, Linearsize, Redshift, precision):
    if np.int(precision) == 0:
        HDF5datatype = 'float32'
        npdatatype = np.float32
        print("Saving in 32bit HDF5 format.")
    if np.int(precision) == 1:
        HDF5datatype = 'double'
        npdatatype = np.float64
        print("Saving in 64bit HDF5 format.")
    N = len(dataarray)
    M_min = np.min(dataarray[:,6])
    HDF5_snapshot = h5py.File(outputfilename, "w")
    #Creating the header
    header_group = HDF5_snapshot.create_group("/Header")
    #Writing the header attributes
    header_group.attrs['NumPart_ThisFile'] = np.array([0,N,0,0,0,0],dtype=np.uint32)
    header_group.attrs['NumPart_Total'] = np.array([0,N,0,0,0,0],dtype=np.uint32)
    header_group.attrs['NumPart_Total_HighWord'] = np.array([0,0,0,0,0,0],dtype=np.uint32)
    header_group.attrs['MassTable'] = np.array([0,M_min,0,0,0,0],dtype=np.float64)
    header_group.attrs['Time'] = np.double(1.0/(Redshift+1))
    header_group.attrs['Redshift'] = np.double(Redshift)
    header_group.attrs['Lbox'] = np.double(Linearsize)
    header_group.attrs['NumFilesPerSnapshot'] = np.int(1)
    header_group.attrs['Omega0'] = np.double(1.0)
    header_group.attrs['OmegaLambda'] = np.double(0.0)
    header_group.attrs['HubbleParam'] = np.double(1.0)
    header_group.attrs['Flag_Sfr'] = np.int(0)
    header_group.attrs['Flag_Cooling'] = np.int(0)
    header_group.attrs['Flag_StellarAge'] = np.int(0)
    header_group.attrs['Flag_Metals'] = np.int(0)
    header_group.attrs['Flag_Feedback'] = np.int(0)
    header_group.attrs['Flag_Entropy_ICs'] = np.int(0)
    #Header created.
    #Creating datasets for the particle data
    particle_group = HDF5_snapshot.create_group("/PartType1")
    X = particle_group.create_dataset("Coordinates", (N,3),dtype=HDF5datatype)
    V = particle_group.create_dataset("Velocities", (N,3),dtype=HDF5datatype)
    IDs = particle_group.create_dataset("ParticleIDs", (N,),dtype='uint64')
    M = particle_group.create_dataset("Masses", (N,),dtype=HDF5datatype)
    #Saving the particle data
    X[:,:] = dataarray[:,0:3]
    V[:,:] = dataarray[:,3:6]
    M[:] = dataarray[:,6]
    IDs[:] = np.arange(N, dtype=np.uint64)
    HDF5_snapshot.close()
    return;
