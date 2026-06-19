import numpy as np
from pynverse import inversefunc

#Defining functions for the constant omega binning
def Calculate_rlimits_i(i, d_s, N_r_bin, last_cell_size):
    r_i = d_s*np.tan((i)*np.pi/(2.0*(N_r_bin+last_cell_size)))
    return r_i;

def Calculate_r_i(i, d_s, N_r_bin, last_cell_size):
    '''
    Calculates the center of the i-th bin for the constant size binning in the
    non-compact space (constant size in the compact space) in 3D StePS geometry.
    Inputs:
        - i = the ID of the boundary
        - d_s = the diameter of the 4D sphere
        - N_r_bin = Number of the radial bins
        - last_cell_size = size of the last, non-infinite cell
    Outputs:
        - r_i = the center of the i-th bin
    '''
    lower_limit = Calculate_rlimits_i(i, d_s, N_r_bin, last_cell_size)
    upper_limit = Calculate_rlimits_i(i+1, d_s, N_r_bin, last_cell_size)
    #Calculating the center of the bin with "conical frustum"
    r_i = 0.25*(upper_limit-lower_limit)*(lower_limit*lower_limit+2*lower_limit*upper_limit+3*upper_limit*upper_limit)/(lower_limit*lower_limit+lower_limit*upper_limit+upper_limit*upper_limit)+lower_limit
    return r_i;

def Calculate_i_r(r, d_s, N_r_bin, last_cell_size):
    i = int(np.arctan(r/d_s)*(2.0*(N_r_bin+last_cell_size))/np.pi)
    return i;

def Mass_resolution_i_3D(i, d_s, N_r_bin, N_part_shell, rho_mean, last_cell_size):
    '''
    i = ID of the shell
    d_s = the diameter of the 4D sphere
    N_r_bin = Number of the radial bins
    N_part_shell = number of particles in the shell
    rho_mean = the aveage density in the Universe
    last_cell_size = size of the last, non-infinite cell
    '''
    #Calculating the mass resolution in 3D StePS geometry
    #for flat cosmology
    R0 = Calculate_rlimits_i(i, d_s, N_r_bin, last_cell_size)
    R1 = Calculate_rlimits_i(i+1, d_s, N_r_bin, last_cell_size)
    Volume = 4.0*np.pi/3.0*((R1)**3 - (R0)**3)
    N_part = N_part_shell
    M_part_out = rho_mean*Volume/N_part #10^11Msol
    return M_part_out;

def Mass_resolution_i_2D(i, Lz, d_s, N_r_bin, N_part_shell, rho_mean, last_cell_size):
    '''
    i = ID of the shell
    Lz = the length of the cylinder in the Z direction
    d_s = the diameter of the 3D sphere
    N_r_bin = Number of the radial bins
    N_part_shell = number of particles in the shell
    rho_mean = the aveage density in the Universe
    last_cell_size = size of the last, non-infinite cell
    '''
    #Calculating the mass resolution in 2D StePS geometry
    #for flat cosmology
    R0 = Calculate_rlimits_i(i, d_s, N_r_bin, last_cell_size)
    R1 = Calculate_rlimits_i(i+1, d_s, N_r_bin, last_cell_size)
    Volume = np.pi*((R1)**2 - (R0)**2)*Lz
    N_part = N_part_shell
    M_part_out = rho_mean*Volume/N_part #10^11Msol
    return M_part_out;

#Defining functions for the constant volume binning (constant volume in the compact space)
def Calculate_rlimits_i_3D_cvol(i, d_s, N_r_bin, R_sim):
    '''
    Calculates the lower limit of the i-th bin for the constant volume binning in the
    non-compact space (constant volume in the compact space) in 3D StePS geometry.

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

def Calculate_rlimits_i_2D_cvol(i, d_s, N_r_bin, R_sim):
    '''
    Calculates the lower limit of the i-th bin for the constant volume binning in the
    non-compact space (constant volume in the compact space) in 3D StePS geometry.

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
    lower_limit = Calculate_rlimits_i_3D_cvol(i, d_s, N_r_bin, R_sim)
    upper_limit = Calculate_rlimits_i_3D_cvol(i+1, d_s, N_r_bin, R_sim)
    #Calculating the center of the bin with "conical frustum"
    r_i = 0.25*(upper_limit-lower_limit)*(lower_limit*lower_limit+2*lower_limit*upper_limit+3*upper_limit*upper_limit)/(lower_limit*lower_limit+lower_limit*upper_limit+upper_limit*upper_limit)+lower_limit
    return r_i;

def Mass_resolution_i_3D_cvol(i, d_s, N_r_bin, N_part_shell, rho_mean, R_sim):
    '''
    i = ID of the shell
    N_part_shell = number of particles in the shell
    rho_mean = the aveage density in the Universe
    R_sim = the radius of the simulation volume in real space
    '''
    #Calculating the mass resolution
    #for flat cosmology
    R0 = Calculate_rlimits_i_3D_cvol(i, d_s, N_r_bin, R_sim)
    R1 = Calculate_rlimits_i_3D_cvol(i+1, d_s, N_r_bin, R_sim)
    Volume = 4.0*np.pi/3.0*((R1)**3 - (R0)**3)
    N_part = N_part_shell
    M_part_out = rho_mean*Volume/N_part #10^11Msol
    return M_part_out;

def Mass_resolution_i_2D_cvol(i, Lz, d_s, N_r_bin, N_part_shell, rho_mean, R_sim):
    '''
    i = ID of the shell
    N_part_shell = number of particles in the shell
    rho_mean = the aveage density in the Universe
    R_sim = the radius of the simulation volume in real space
    '''
    #Calculating the mass resolution
    #for flat cosmology
    R0 = Calculate_rlimits_i_2D_cvol(i, d_s, N_r_bin, R_sim)
    R1 = Calculate_rlimits_i_2D_cvol(i+1, d_s, N_r_bin, R_sim)
    Volume = np.pi*((R1)**2 - (R0)**2)*Lz
    N_part = N_part_shell
    M_part_out = rho_mean*Volume/N_part #10^11Msol
    return M_part_out;