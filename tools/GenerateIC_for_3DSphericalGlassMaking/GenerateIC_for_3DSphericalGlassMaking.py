#!/usr/bin/env python3

import numpy as np
import healpy as hp
from pynverse import inversefunc
import yaml
import sys
import time

#Defining functions for the constant omega binning
def Calculate_rlimits_i(i, d_s, N_r_bin, last_cell_size):
    r_i = d_s*np.tan((i)*np.pi/(2.0*(N_r_bin+last_cell_size)))
    return r_i;

def Calculate_r_i(i, d_s, N_r_bin, last_cell_size):
    lower_limit = Calculate_rlimits_i(i, d_s, N_r_bin, last_cell_size)
    upper_limit = Calculate_rlimits_i(i+1, d_s, N_r_bin, last_cell_size)
    #Calculating the center of the bin with "conical frustum"
    r_i = 0.25*(upper_limit-lower_limit)*(lower_limit*lower_limit+2*lower_limit*upper_limit+3*upper_limit*upper_limit)/(lower_limit*lower_limit+lower_limit*upper_limit+upper_limit*upper_limit)+lower_limit
    return r_i;

def Calculate_i_r(r, d_s, N_r_bin, last_cell_size):
    i = int(np.arctan(r/d_s)*(2.0*(N_r_bin+last_cell_size))/np.pi)
    return i;

def Mass_resolution_i(i, d_s, N_r_bin, N_part_shell, rho_mean, last_cell_size):
    '''
    i = ID of the shell
    N_part_shell = number of particles in the shell
    rho_mean = the aveage density in the Universe
    last_cell_size = size of the last, non-infinite cell
    '''
    #Calculating the mass resolution
    #for flat cosmology
    R0 = Calculate_rlimits_i(i, d_s, N_r_bin, last_cell_size)
    R1 = Calculate_rlimits_i(i+1, d_s, N_r_bin, last_cell_size)
    Volume = 4.0*np.pi/3.0*((R1)**3 - (R0)**3)
    N_part = N_part_shell
    M_part_out = rho_mean*Volume/N_part #10^11Msol
    return M_part_out;

#Defining functions for the constant volume binning (constant volume in the compact space)
def Calculate_rlimits_i_cvol(i, d_s, N_r_bin, R_sim):
    '''
    Calculates the lower limit of the i-th bin for the constant volume binning in the
    non-compact space (constant volume in the compact space)

    i = the ID of the boundary
    d_s = the diameter of the 4D sphere
    N_r_bin = Number of the radial bins
    R_sim = the radius of the simulation volume in real space
    '''
    omega_max = 2.0*np.arctan(R_sim/d_s)
    V_unit_bin = (2.0*omega_max-np.sin(2.0*omega_max))/N_r_bin
    V_unit_to_i = i*V_unit_bin
    #inverting numerically the x-sin(x) function
    func = (lambda x: x-np.sin(x))
    omega_i = inversefunc(func, y_values=V_unit_to_i)/2.0
    r_i = d_s*np.tan(omega_i/2)
    return r_i;

def Calculate_r_i_cvol(i, d_s, N_r_bin, R_sim):
    lower_limit = Calculate_rlimits_i_cvol(i, d_s, N_r_bin, R_sim)
    upper_limit = Calculate_rlimits_i_cvol(i+1, d_s, N_r_bin, R_sim)
    #Calculating the center of the bin with "conical frustum"
    r_i = 0.25*(upper_limit-lower_limit)*(lower_limit*lower_limit+2*lower_limit*upper_limit+3*upper_limit*upper_limit)/(lower_limit*lower_limit+lower_limit*upper_limit+upper_limit*upper_limit)+lower_limit
    return r_i;

def Mass_resolution_i_cvol(i, d_s, N_r_bin, N_part_shell, rho_mean, R_sim):
    '''
    i = ID of the shell
    N_part_shell = number of particles in the shell
    rho_mean = the aveage density in the Universe
    R_sim = the radius of the simulation volume in real space
    '''
    #Calculating the mass resolution
    #for flat cosmology
    R0 = Calculate_rlimits_i_cvol(i, d_s, N_r_bin, R_sim)
    R1 = Calculate_rlimits_i_cvol(i+1, d_s, N_r_bin, R_sim)
    Volume = 4.0*np.pi/3.0*((R1)**3 - (R0)**3)
    N_part = N_part_shell
    M_part_out = rho_mean*Volume/N_part #10^11Msol
    return M_part_out;

def Generate_random_shell(Nshell):
    #Generating a spherical shell with random Nshell particles, and with 1 radius
    shell = np.zeros((Nshell,3))
    for i in range(0, Nshell):
        while True:
            shell[i,:] = np.random.rand(3)-0.5
            radius = np.sqrt(shell[i,0]**2 + shell[i,1]**2 +shell[i,2]**2)
            if radius <= 0.5:
                shell[i,:] /= radius
                break
    return shell;


#Begininng of the script
print("----------------------------------------------------------------------------------------------\nGenerateIC_for_3DSphericalGlassMaking v0.2.1.1\n (A 3D initial condition generator for glass making.)\n\n Gabor Racz, 2018\n\tDepartment of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary\n\tDepartment of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA\n----------------------------------------------------------------------------------------------\n\n")

#read the input parameters (from the first argument)
if len(sys.argv) != 2:
        print("usage: ./GenerateIC_for_3DSphericalGlassMaking.py <input yaml file>")
        sys.exit(2)
start = time.time()
print("Reading the %s paramfile...\n" % str(sys.argv[1]))
document = open(str(sys.argv[1]))
Params = yaml.load(document)
print("Glass parameters:\n----------------------\nOutput file:\t\t\t\t%s\nDiameter of the 4D hypersphere:\t\t%fMpc\nRadius of the simulation volume:\t%fMpc\nRandom seed:\t\t\t\t%i" % (Params['BASEOUT'], Params['D_S'], Params['RSIM'], Params['RANDSEED'] ))
np.random.seed(Params['RANDSEED'])
if Params['BIN_MODE'] == 0:
    print("Binning mode:\t\t\t\tConstant size binning in the \"omega\" compact coordinate.")
    print("Input Periodic Glass:\t\t\t%s" % Params['GLASSFILE'])
    print("Radius of constant resolution:\t\t%fMpc" % Params['RCRIT'] )
    last_cell_size = Params['NRBINS']*np.pi/(2*np.arctan(Params['RSIM']/Params['D_S']))-Params['NRBINS']
if Params['BIN_MODE'] == 1:
    print("Binning mode:\t\t\t\tConstant shell volumes in the compact space.")
if (Params['BIN_MODE'] > 1) or (Params['BIN_MODE'] < 0):
    print("Error: Binning mode = %i\n Unknown binning mode!\n Exiting." % Params['BIN_MODE'])
    sys.exit(2)
#importing the matplotlib, if MAKEPLOTS is on
if Params['MAKEPLOTS'] == 1:
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
print("Number of particles per shell:\t\t%i\nNumber of radial bins:\t\t\t%i\n\n" % (Params['NSHELL'], Params['NRBINS']))
print("Cosmological parameters:\n------------------------\nOmega_lambda\t%f\nOmega_m\t\t%f\nOmega_k\t\t%f\nH0\t\t%f(km/s)/Mpc\n" % (Params['OMEGA_L'], Params['OMEGA_M'], 1.0-Params['OMEGA_M']-Params['OMEGA_L'], Params['HUBBLE_CONSTANT']))
#Calculating the mean density:
rho_crit = 3*Params['HUBBLE_CONSTANT']**2/(8*np.pi)*0.0482191394711204*0.0482191394711204 #G=1
rho_mean = rho_crit*Params['OMEGA_M']
#Calculating the resolutions:
if Params['BIN_MODE'] == 0:
    i_crit = Calculate_i_r(Params['RCRIT'], Params['D_S'], Params['NRBINS'], last_cell_size)
    i = np.arange(Params['NRBINS'])
    r = Calculate_r_i(i, Params['D_S'], Params['NRBINS'], last_cell_size)
    Mass = np.zeros(Params['NRBINS'])
    Mass_res_inside = Mass_resolution_i(i_crit, Params['D_S'], Params['NRBINS'], Params['NSHELL'], rho_mean, last_cell_size)
    #Calculating the total number of particles:
    N_part_inside = int((4.0*np.pi/3.0*Calculate_rlimits_i(i_crit, Params['D_S'], Params['NRBINS'], last_cell_size)**3)*rho_mean/Mass_res_inside)
    #recalculating the mass resolution inside RCRIT
    Mass_res_inside = ((4.0*np.pi/3.0*Calculate_rlimits_i(i_crit, Params['D_S'], Params['NRBINS'], last_cell_size)**3)*rho_mean/N_part_inside)
    N_part_outside = (Params['NRBINS']-i_crit)*Params['NSHELL']
    N_part = N_part_inside + N_part_outside
    #Allocating memory for the particles
    print("Total number of particles =\t\t%i\n" % N_part)
    print("Number of particles inside the constant resolution region =\t%i\n" % N_part_inside)
    print("Number of particles outside the constant resolution region =\t%i\n" % N_part_outside)
    print("The Mass resolution inside the constant resolution region = %lf 10e11Msol\n" % Mass_res_inside)
    for j in i:
        if j>=i_crit:
            Mass[j] = Mass_resolution_i(j, Params['D_S'], Params['NRBINS'], Params['NSHELL'], rho_mean, last_cell_size)
        else:
            Mass[j] = Mass_res_inside
if Params['BIN_MODE'] == 1:
    i = np.arange(Params['NRBINS'])
    N_part = np.int64(Params['NRBINS'])*np.int64(Params['NSHELL'])
    r = Calculate_r_i_cvol(i, Params['D_S'], Params['NRBINS'], Params['RSIM'])
    Mass = Mass_resolution_i_cvol(i, Params['D_S'], Params['NRBINS'], Params['NSHELL'], rho_mean, Params['RSIM'])
    print("Total number of particles =\t\t%i\n" % N_part)
    print("The Mass resolution at the tangent point = %lf 10e11Msol\n" % Mass[0])
    print("The calculated real-space bins:\nr_min\t\tr_i\t\tr_max\t\tMass_part")
    print(np.array(( Calculate_rlimits_i_cvol(i, Params['D_S'], Params['NRBINS'], Params['RSIM']),r, Calculate_rlimits_i_cvol(i+1, Params['D_S'], Params['NRBINS'], Params['RSIM']),Mass)).T)

particle_data = np.zeros((N_part,7)) #x, y, z, vx, vy, vz, Mass
if Params['MAKEPLOTS'] == 1:
    Mill_res = 0.86/(73.0/100.0)/100 + 0*i
    plt.figure(figsize=(10, 8))
    plt.xlabel(r'$r[Mpc]$')
    plt.ylabel(r'$M[10^{11} M_{\odot}]$')
    axes = plt.gca()
    axes.set_xlim([0.0,Params['RSIM']])
    plt.grid()
    plt.semilogy(r,Mass, label="StePS Resolution")
    plt.semilogy(r,Mill_res, label="Millennium Resolution")
    plt.legend()
    plt.title("Resolution, as a function of radius")
    plt.show()
if Params['BIN_MODE'] == 0:
    #Generating the r>RCRIT range:
    for j in range(i_crit,Params['NRBINS']):
        #Generating a random shell
        spherical_glass = Generate_random_shell(Params['NSHELL'])
        #the thickness of the shell
        D=Calculate_rlimits_i(j+1, Params['D_S'], Params['NRBINS'], last_cell_size)-Calculate_rlimits_i(j, Params['D_S'], Params['NRBINS'], last_cell_size)
        radii = (Calculate_rlimits_i(j+1, Params['D_S'], Params['NRBINS'], last_cell_size)+Calculate_rlimits_i(j, Params['D_S'], Params['NRBINS'], last_cell_size))*0.5
        for i in range(0,Params['NSHELL']):
            #setting the radii of the shells with a small random noise
            particle_data[(N_part_inside+(j-i_crit)*Params['NSHELL']+i),0:3] = (radii+D*(np.random.rand()-0.5))*spherical_glass[i,0:3]
        particle_data[(N_part_inside+(j-i_crit)*Params['NSHELL']):(N_part_inside+(j-i_crit+1)*Params['NSHELL']),6] = Mass[j]
    del(spherical_glass)
if Params['BIN_MODE'] == 1:
    #Generating the initial random coordinates
    for j in range(0,Params['NRBINS']):
        #Generating a random shell
        spherical_glass = Generate_random_shell(Params['NSHELL'])
        D=Calculate_rlimits_i_cvol(j+1, Params['D_S'], Params['NRBINS'],  Params['RSIM'])-Calculate_rlimits_i_cvol(j, Params['D_S'], Params['NRBINS'],  Params['RSIM'])
        r_j_limit_shell = Calculate_rlimits_i_cvol(j, Params['D_S'], Params['NRBINS'],  Params['RSIM'])
        for i in range(0,Params['NSHELL']):
            #setting the radii of the shells with a small random displacement in the shell
            displacement = np.cbrt(np.random.rand()*((r_j_limit_shell+D)**3 - r_j_limit_shell**3)+r_j_limit_shell**3)-r_j_limit_shell
            particle_data[(j*Params['NSHELL']+i),0:3] = (r_j_limit_shell+displacement)*spherical_glass[i,0:3]
            particle_data[(j*Params['NSHELL']+i),6] = Mass[j]
if Params['BIN_MODE'] == 0:
    #Generating the r<RCRIT range from the input periodic glass
    print("Reading the %s input periodic glass..." % Params['GLASSFILE'])
    periodic_glass = np.fromfile(Params['GLASSFILE'], count=-1, sep='\t', dtype=np.float64)
    periodic_glass = periodic_glass.reshape(int(len(periodic_glass)/6),6)
    L = np.max(periodic_glass)
    #deleting the velocities
    periodic_glass = np.delete(periodic_glass, [4,5], axis=1)
    print("...done\n")
    if len(periodic_glass)<N_part_inside:
        print("Error: The input periodic glass do not contains enough particles!\nUse bigger input glass, or try to use newer version of this script if available!\nExiting...\n")
        sys.exit()
    #Shifting the glass to the center
    periodic_glass = periodic_glass-L/2
    #Calculating r for every particle
    periodic_glass[:,3] = np.sqrt(periodic_glass[:,0]**2 + periodic_glass[:,1]**2 + periodic_glass[:,2]**2 )
    #sorting the periodic glass
    print("Sorting the data...")
    ind = np.argsort(periodic_glass[:,3])
    periodic_glass = np.array([periodic_glass[j,:] for j in ind])
    print("...done\n")
    #Cutting off the corners of the periodic box
    end_index = np.searchsorted(periodic_glass[:,3], L/2, 'right')
    periodic_glass = np.delete(periodic_glass, np.array(range(end_index,len(periodic_glass))), axis=0)
    if end_index<N_part_inside:
        print("Error: The input periodic glass do not contains enough particles!\nUse bigger input glass, or try to use newer version of this script if available!\nExiting...\n")
        sys.exit()
    #Cutting out the particles we only need
    periodic_glass = np.delete(periodic_glass, np.array(range(N_part_inside,end_index+1)), axis=0)
    #Rescaling the sphere
    periodic_glass = periodic_glass / periodic_glass[N_part_inside-1,3] * Calculate_rlimits_i(i_crit-0.25, Params['D_S'], Params['NRBINS'], last_cell_size)
    #copying the particles inside the constant resolution region
    particle_data[0:N_part_inside,0:3] = periodic_glass[:,0:3]
    particle_data[0:N_part_inside,6] = Mass_res_inside
if Params['MAKEPLOTS'] == 1:
    print("Making plot...")
    alpha = 2.5/180.0*np.pi #2.5*2 degree viewing angle in RAD
    N_slice = 0
    for i in range(0,N_part):
        if particle_data[i,2]>-alpha*np.sqrt(particle_data[i,0]**2 + particle_data[i,1]**2) and particle_data[i,2]<alpha*np.sqrt(particle_data[i,0]**2 + particle_data[i,1]**2):
                    N_slice+=1
    slice_for_plot = np.zeros( (N_slice,3), dtype=np.float64)
    j=0
    for i in range(0,N_part):
        if particle_data[i,2]>-alpha*np.sqrt(particle_data[i,0]**2 + particle_data[i,1]**2) and particle_data[i,2]<alpha*np.sqrt(particle_data[i,0]**2 + particle_data[i,1]**2):
            slice_for_plot[j,0] = particle_data[i,0]
            slice_for_plot[j,1] = particle_data[i,1]
            slice_for_plot[j,2] = particle_data[i,6]
            j+=1
    plt.figure(figsize=(6,6))
    axes = plt.gca()
    if Params['BIN_MODE'] == 0:
        axes.set_xlim([-Params['RCRIT']*1.5,Params['RCRIT']*1.5])
        axes.set_ylim([-Params['RCRIT']*1.5,Params['RCRIT']*1.5])
    if Params['BIN_MODE'] == 1:
        axes.set_xlim([-Params['D_S']*1.5,Params['D_S']*1.5])
        axes.set_ylim([-Params['D_S']*1.5,Params['D_S']*1.5])
    plt.title("The generated particles around the constant resolution region")
    if Params['BIN_MODE'] == 0:
        plt.scatter(slice_for_plot[:,0], slice_for_plot[:,1], marker='o', c='b', s=np.sqrt(slice_for_plot[:,2]/Mass_res_inside)/4.0)
    if Params['BIN_MODE'] == 1:
        plt.scatter(slice_for_plot[:,0], slice_for_plot[:,1], marker='o', c='b', s=np.sqrt(slice_for_plot[:,2]/Mass[0])/4.0)
    plt.xlabel('x[Mpc]'); plt.ylabel('y[Mpc]'); plt.grid()
    print("...done")
    plt.show()
#calculating the total mass
Tot_mass = np.sum(particle_data[:,6])
#the simulation volume is:
Tot_V = 4*np.pi/3*Params['RSIM']**3
#the average density (in Omega_m)
Avg_dens = (Tot_mass/Tot_V)/rho_mean
print("The total Mass = %e 10e11M_sol\nThe volume = %eMpc^3\nThe mean density = %f" % (Tot_mass, Tot_V, Avg_dens))
print("Saving...")
np.savetxt(str(Params['BASEOUT']), particle_data, delimiter='\t')
end = time.time()
if Params['MAKEPLOTS'] == 0:
    print("...done.\n The generation of the IC for glass making took %fs" % (end-start))
else:
    print("...done.")
