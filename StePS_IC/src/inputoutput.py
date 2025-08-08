#!/usr/bin/python3

#*******************************************************************************#
#  StePS_IC.py - An initial condition generator for                             #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2017-2024 Gabor Racz                                         #
#                                                                               #
#    This program is free software; you can redistribute it and/or modify       #
#    it under the terms of the GNU General Public License as published by       #
#    the Free Software Foundation; either version 2 of the License, or          #
#    (at your option) any later version.                                        #
#                                                                               #
#    This program is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#    GNU General Public License for more details.                               #
#*******************************************************************************#

# Functions for reading and writing particle data

import sys
import time
from pygadgetreader import *
from past.translation import autotranslate
autotranslate(['glio'])
import glio
import numpy as np
import h5py
from astropy.units import solMass,Mpc,m,s

#defining functions

def Load_snapshot(FILENAME,CONSTANT_RES=False,RETURN_VELOCITIES=False,RETURN_IDs=False,SILENT=False,DOUBLEPRECISION=False):
    if DOUBLEPRECISION:
        floatdtype=np.float64
    else:
        floatdtype=np.float32
    #reading the input files:
    if FILENAME[-4:] == '.dat':
        if SILENT==False:
            print("\tThe input file is in ASCII format. Reading the input file %s ..." % (FILENAME))
        data = np.loadtxt(FILENAME)
        Coordinates = data[:,0:3]
        if RETURN_VELOCITIES:
            Velocities = data[:,3:6]
        Masses = data[:,6]
        if RETURN_IDs:
            ParticleIDs = np.arange(len(Masses), dtype=np.uint64)
        del(data)
        if SILENT==False:
            print("\t...done\n")
    elif FILENAME[-4:] == 'hdf5':
        if SILENT==False:
            print("\tThe input file is in hdf5 format.")
        if FILENAME[-7:] == '.0.hdf5':
            if SILENT==False:
                print("\tSnapshot is stored in multiple files.")
            fileindex = 0
            while True:
                filename = list(FILENAME)
                filename[-7:] = ".%i.hdf5"%fileindex
                filename_str = "".join(filename)
                if os.path.exists(filename_str):
                    if SILENT==False:
                        print("\t\tOpening %s..."%filename_str)
                    HDF5_snapshot = h5py.File(filename_str, "r")
                    if fileindex==0:
                        N = int(HDF5_snapshot['/Header'].attrs['NumPart_Total'][1])
                        Nthisfile = int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1])
                        Coordinates = np.zeros((N,3), dtype=floatdtype)
                        if RETURN_VELOCITIES:
                            Velocities = np.zeros((N,3), dtype=floatdtype)
                        if RETURN_IDs:
                            ParticleIDs = np.zeros(N, dtype=np.uint64)
                        Masses = np.ones((N), dtype=floatdtype)
                        Coordinates[:Nthisfile,0:3] = HDF5_snapshot['/PartType1/Coordinates']
                        if RETURN_VELOCITIES:
                            Velocities[:Nthisfile,0:3] = HDF5_snapshot['/PartType1/Velocities']
                        if CONSTANT_RES==False:
                            Masses[:Nthisfile] = HDF5_snapshot['/PartType1/Masses']
                        else:
                            Masses *= HDF5_snapshot['/Header'].attrs['MassTable'][1] #1e11Msol
                        if RETURN_IDs:
                            ParticleIDs[:Nthisfile] = HDF5_snapshot['/PartType1/ParticleIDs']
                        Nfilled=Nthisfile
                    else:
                        Nthisfile = int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1])
                        Coordinates[Nfilled:Nfilled+Nthisfile,0:3] = HDF5_snapshot['/PartType1/Coordinates']
                        if RETURN_VELOCITIES:
                            Velocities[Nfilled:Nfilled+Nthisfile,0:3] = HDF5_snapshot['/PartType1/Velocities']
                        if RETURN_IDs:
                            ParticleIDs[Nfilled:Nfilled+Nthisfile] = HDF5_snapshot['/PartType1/ParticleIDs']
                        if CONSTANT_RES==False:
                            Masses[Nfilled:Nfilled+Nthisfile] = HDF5_snapshot['/PartType1/Masses']
                        Nfilled = Nfilled+Nthisfile
                    HDF5_snapshot.close()
                    del(HDF5_snapshot)
                    fileindex+=1
                    if SILENT==False:
                        print("\t\t...done.")
                else:
                    if SILENT==False:
                        print("\tAll file loaded for snapshot %s."%FILENAME)
                    break;
        else:
            if SILENT==False:
                print("\tReading the input file %s ..." % (FILENAME))
            HDF5_snapshot = h5py.File(FILENAME, "r")
            N = int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1])
            Coordinates = np.zeros((N,3), dtype=floatdtype)
            if RETURN_VELOCITIES:
                Velocities = np.zeros((N,3), dtype=floatdtype)
            if RETURN_IDs:
                ParticleIDs = np.zeros((N), dtype=np.uint64)
            Masses = np.zeros((N), dtype=floatdtype)
            Coordinates[:,0:3] = HDF5_snapshot['/PartType1/Coordinates']
            if RETURN_VELOCITIES:
                Velocities[:,0:3] = HDF5_snapshot['/PartType1/Velocities']
            if CONSTANT_RES==False:
                Masses[:] = HDF5_snapshot['/PartType1/Masses']
            if RETURN_IDs:
                ParticleIDs[:] = HDF5_snapshot['/PartType1/ParticleIDs']
            HDF5_snapshot.close()
            del(HDF5_snapshot)
            if SILENT==False:
                print("\t...done\n")
    else:
        if SILENT==False:
            print("\tAssuming gadget-format for the inputfile. Trying to open it...")
        Coordinates = readsnap(FILENAME, 'pos', 'dm')
        Masses = readsnap(FILENAME, 'mass', 'dm')
        if SILENT==False:
            print('\tMasses =%f' %np.mean(Masses))
            print("\t...done.")

    if RETURN_VELOCITIES==False:
        if RETURN_IDs:
            return Coordinates, Masses, ParticleIDs
        else:
            return Coordinates, Masses
    else:
        if RETURN_IDs:
            return Coordinates, Velocities, Masses, ParticleIDs
        else:
            return Coordinates, Velocities, Masses

def Load_params_from_HDF5_snap(FILENAME):
    if FILENAME[-4:] != 'hdf5' and FILENAME[-4:] != 'HDF5':
        raise Exception("Error: input file %s is not in hdf5 format.\n" % FILENAME)
    HDF5_snapshot = h5py.File(FILENAME, "r")
    Ntot = int(HDF5_snapshot['/Header'].attrs['NumPart_Total'][1])
    z = np.double(HDF5_snapshot['/Header'].attrs['Redshift'])
    Om = np.double(HDF5_snapshot['/Header'].attrs['Omega0'])
    Ol = np.double(HDF5_snapshot['/Header'].attrs['OmegaLambda'])
    H0 = np.double(HDF5_snapshot['/Header'].attrs['HubbleParam'])*100.0
    HDF5_snapshot.close()
    return z, Om, Ol, H0, Ntot

def writeHDF5snapshot(dataarray, outputfilename, Linearsize, Redshift, OmegaM, OmegaL, HubbleParam, precision):
    '''
    Function for writing out IC in hdf5 format.
    Parameters:
        dataarray - numpy array containing the particle data (coordinates, velocities, masses)
        outputfilename - name of the output file
        Linearsize - Linear size of the IC (simulation)
        Redshift - initial redshift
        OmegaM - Matter density
        OmegaL - Dark energy density
        HubbleParam - Hubble parameter
        precision - Floating point precision of the IC (0: 32bit; 1: 64bit)
    '''
    if int(precision) == 0:
        HDF5datatype = 'float32'
        npdatatype = np.float32
        print("Saving in 32bit HDF5 format.")
    if int(precision) == 1:
        HDF5datatype = 'double'
        npdatatype = np.float64
        print("Saving in 64bit HDF5 format.")
    N = len(dataarray)
    HDF5_snapshot = h5py.File(outputfilename, "w")
    #Creating the header
    header_group = HDF5_snapshot.create_group("/Header")
    #Writing the header attributes
    header_group.attrs['NumPart_ThisFile'] = np.array([0,N,0,0,0,0],dtype=np.uint32)
    header_group.attrs['NumPart_Total'] = np.array([0,N,0,0,0,0],dtype=np.uint32)
    header_group.attrs['NumPart_Total_HighWord'] = np.array([0,0,0,0,0,0],dtype=np.uint32)
    header_group.attrs['MassTable'] = np.array([0,0,0,0,0,0],dtype=npdatatype)
    header_group.attrs['Time'] = np.double(1.0/(Redshift+1))
    header_group.attrs['Redshift'] = np.double(Redshift)
    header_group.attrs['BoxSize'] = np.double(Linearsize)
    header_group.attrs['NumFilesPerSnapshot'] = int(1)
    header_group.attrs['Omega0'] = np.double(OmegaM)
    header_group.attrs['OmegaLambda'] = np.double(OmegaL)
    header_group.attrs['HubbleParam'] = np.double(HubbleParam)
    header_group.attrs['Flag_Sfr'] = int(0)
    header_group.attrs['Flag_Cooling'] = int(0)
    header_group.attrs['Flag_StellarAge'] = int(0)
    header_group.attrs['Flag_Metals'] = int(0)
    header_group.attrs['Flag_Feedback'] = int(0)
    header_group.attrs['Flag_Entropy_ICs'] = int(0)
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

def ascii2gadget(infile, outfile, Lbox, H0, UNITLENGTH_IN_CM):
    '''
    Function to convert a StePS ascii file to Gadget format.
    infile: input StePS ascii file
    outfile: output Gadget file
    '''
    #Setting up the units of distance and time
    UNIT_T=47.14829951063323 #Unit time in Gy
    UNIT_V=20.738652969925447 #Unit velocity in km/s
    UNIT_D=3.0856775814671917e24#=1Mpc Unit distance in cm (in the StePS code)
    #Reading the input data
    particle_data = np.fromfile(infile, count=-1, sep='\t', dtype=np.float64)
    particle_data = particle_data.reshape(int(len(particle_data)/7),7)
    h = H0/100.0
    #Creating array of X coordinates and V velocities
    X = particle_data[:,0:3] * h * UNIT_D / UNITLENGTH_IN_CM
    V = particle_data[:,3:6] #velocities
    M = particle_data[:,6] # Masses
    Npart = len(X)
    del(particle_data)
    #Creating the Gadget-snapshot
    Gadget_snapshot = glio.GadgetSnapshot(outfile)
    Gadget_snapshot.header.npart = np.array([0,Npart,0,0,0,0], dtype=np.int32)
    Gadget_snapshot.header.mass = np.array([0.0,1.0,0.0,0.0,0.0,0.0], dtype=np.float64)
    Gadget_snapshot.header.time= np.array([0.0078125], dtype=np.float64)
    Gadget_snapshot.header.redshift= np.array([127.0], dtype=np.float64)
    Gadget_snapshot.header.flag_sfr= np.array([0], dtype=np.int32)
    Gadget_snapshot.header.flag_feedback= np.array([0], dtype=np.int32)
    Gadget_snapshot.header.npartTotal =  np.array([0,Npart,0,0,0,0], dtype=np.int32)
    Gadget_snapshot.header.flag_cooling = np.array([0], dtype=np.int32)
    Gadget_snapshot.header.num_files = np.array([1], dtype=np.int32)
    Gadget_snapshot.header.BoxSize = np.array([Lbox*h*UNIT_D/UNITLENGTH_IN_CM], dtype=np.float64)
    Gadget_snapshot.header.Omega0 =  np.array([1.0], dtype=np.float64)
    Gadget_snapshot.header.OmegaLambda =  np.array([0.0], dtype=np.float64)
    Gadget_snapshot.header.HubbleParam = np.array([h], dtype=np.float64)
    Gadget_snapshot.header.flag_stellarage = np.array([0], dtype=np.int32)
    Gadget_snapshot.header.flag_metals = np.array([0], dtype=np.int32)
    Gadget_snapshot.header.npartTotalHighWord = np.array([0,0,0,0,0,0], dtype=np.uint32)
    Gadget_snapshot.header.flag_entropy_instead_u = np.array([0], dtype=np.int32)
    Gadget_snapshot.header._padding = np.zeros(15,dtype=np.int32)
    Gadget_snapshot.ID[1] = np.array(range(0,Npart), dtype=np.uint32)
    Gadget_snapshot.pos[1] = np.array(X, dtype=np.float32)
    Gadget_snapshot.vel[1] = np.array(V, dtype=np.float32)
    Gadget_snapshot.save(outfile)
    del(X)
    del(V)
    del(M)
    return;
