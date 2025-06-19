import numpy as np
from scipy.optimize import fsolve
import astropy.units as u
from astropy.cosmology import LambdaCDM, wCDM, w0waCDM
import time
from StePS_HF.geometry import *
from StePS_HF.octree import compute_potentials_tree, cubic_spline_potential
from StePS_HF import G

def get_center_of_mass(r, m, boundaries="STEPS", boxsize=0):
    """
    Function for calculating the center of mass of a particle system
    Input:
        - r: particle coordinates
        - m: particle masses
        - boundaries: boundary condition. Must be "STEPS", "PERIODIC", or "CYLINDRICAL"
        - boxsize: linear box size in the same units as the coorinates
    Returns:
        - Center of mass coordinates
    """
    if boundaries=="STEPS":
        return np.sum((r.T*m).T,axis=0)/np.sum(m)
    elif boundaries=="PERIODIC":
        #assuming that the box size is significantly larger than the halo size (moving the first particle to the center)
        dr = boxsize/2 - r[0,:]
        r =  np.mod(r+dr,boxsize)
        com = np.sum((r.T*m).T,axis=0)/np.sum(m)
        return com-dr# shifting the center back to the original position and returning the center of mass
    elif boundaries=="CYLINDRICAL":
        # assuming that the periodicity in z direction is significantly larger than the halo size
        dr = boxsize/2 - r[0,2]
        r[:,2] = np.mod(r[:,2]+dr,boxsize)
        com = np.sum((r.T*m).T,axis=0)/np.sum(m)
        com[2] -= dr # shifting the center back to the original position and returning the center of mass
        return com
    else:
        raise Exception("Error: unknown boundary condition %s." % (boundaries))

def get_angular_momentum(r,v,m,a,hubblenow):
    """
    Function for calculating the angular momentum of a particle system.
    Assumed input:
        - r: CoM comoving coordinates.
        - v: comoving velocities.
        - m: particle masses.
        - a: scale factor
        - hubblenow: Hubble parameter at the current scale factor in km/s/Mpc units
    Returns:
        - J: angular momentum vector in physical units
    """
    p = m.reshape((len(m),1))*(v+hubblenow*a*r) #Individual linear momenta: mass x (comoving) velocity
    J = np.cross(r*a,p)# Individual orbital angular momenta" (comoving position vector) x (linear momentum)
    return np.sum(J,axis=0) #returning the total angular momentum vector

def get_individual_energy(r,v,m,a,Hubble,force_res,boundary="STEPS",boxsize=0.0, method="direct"):
    """
    Function for calculating the individual energy of a particle system by using direct summation of the potential.
    Notes:
        * Kinetic energy is calculated by using the formula Ekin = \sum_i 0.5*m*v_{i, physical}^2 = 0.5*m*(v_{i, comoving}+v_{i,Hubble})^2
          calculated from halo center of mass velocity and Hubble expansion.
        * Potential energy is calculated by using the cubic spline potential (Monaghan & Lattanzio, 1985)
    Expected input:
        - r: CoM coordinates in Mpc.
        - v: velocities in km/s.
        - m: particle masses in Msol.
        - a: scale factor
        - Hubble: Hubble parameter at "a" scale factor in km/s/Mpc units
        - force_res: softening length of each particle
        - boundary: boundary condition. Must be "STEPS", "PERIODIC", or "CYLINDRICAL"
        - boxsize: linear box size in the same units as the coordinates, if periodic or cylindrical boundary conditions are used
        - method: method for calculating the potential. Currently only "direct" and "octree" is implemented
    Returns:
        - Ekin: Kinetic energy in (Msol * (km/s)^2 ) units
        - Epot: Potential energy in (Msol * (km/s)^2 ) units
    """
    Nparticle = len(m) #number of input particles
    Ekin = 0.5*m*np.sum((v + r*a*Hubble)**2,axis=1) # kinetic energy of the individual particles (Ekin = 0.5*m*v_physical^2)
    #calculating the potential energy
    Epot = np.zeros(Nparticle,dtype=np.double)
    if method == "direct":
        for i in range(0,Nparticle):
            idx = np.where(np.arange(0,Nparticle)!=i)
            if boundary == "STEPS" or boundary == "CYLINDRICAL":
                dist = np.sqrt(np.sum(( r[idx] - r[i])**2, axis=1))
            elif boundary == "PERIODIC":
                dist = get_periodic_distances(r[idx], r[i], boxsize)
            else:
                raise Exception("Error: Unknown boundary condition %s." % (boundary))
            Epot[i] += m[i]*np.sum(m[idx]*cubic_spline_potential(dist,force_res[idx]+force_res[i]))
    elif method == "octree":
        #note: the octree is only faster for large particle numbers (Npart > ~20000)
        Epot = compute_potentials_tree(r, m, force_res, theta=0.7, eta=1.0, max_leaf=1)
    else:
        raise Exception("Error: Unknown potential calculation method %s." % (method))
    Epot *= G/a
    return Ekin, Epot

def get_total_energy(r,v,m,a,Hubble,force_res,boundary="STEPS",boxsize=0.0, method="direct"):
        """
        Function for calculating the total energy of a particle system.
        Expected input:
            - r: CoM coordinates in Mpc.
            - v: velocities in km/s.
            - m: particle masses in Msol.
            - a: scale factor
            - Hubble: Hubble parameter at "a" scale factor in km/s/Mpc units
            - force_res: softening length of each particle
            - boundary: boundary condition. Must be "STEPS" or "PERIODIC"
            - boxsize: linear box size in the same units as the coorinates
            - method: method for calculating the potential. Currently only "direct" and "octree" is implemented
        Returns:
            - TotE: Total energy in of the system in (Msol * (km/s)^2 ) units
            - TotEkin: Total kinetic energy of the system in (Msol * (km/s)^2 ) units
            - TotEpot: Total potential energy of the system in (Msol * (km/s)^2 ) units
        """
        Ekin,Epot = get_individual_energy(r,v,m,a,Hubble,force_res,boundary=boundary,boxsize=boxsize, method=method)
        TotEkin = np.sum(Ekin) #total kinetic energy of the halo
        TotEpot = np.sum(Epot) #total potential energy of the halo
        return TotEkin+TotEpot, TotEkin, TotEpot

def get_Rs_Klypin(vmax,v200,R200):
    """
    Function for calculatin the c concentration and Rs scale length based on Klypin Vmax method.
    Input:
        - vmax: maximal circular velocity of the halo
        - v200: circular velocity at the virial radius
        - R200: virial radius
    Returns:
        - c: Rvir/Rs concentration of the halo
        - Rs: Scale radius of the halo
    Details:
        -> Francisco Prada, Anatoly A. Klypin, Antonio J. Cuesta, Juan E. Betancort-Rijo, Joel Primack (2012) https://academic.oup.com/mnras/article/423/4/3018/987360
        -> Klypin, Anatoly A. ; Trujillo-Gomez, Sebastian ; Primack, Joel (2011) https://ui.adsabs.harvard.edu/abs/2011ApJ...740..102K/abstract
        -> Klypin, Anatoly ; Kravtsov, Andrey V. ; Bullock, James S. ; Primack, Joel R. (2001) https://ui.adsabs.harvard.edu/abs/2001ApJ...554..903K/abstract
    """
    vmaxperv200sqr = (vmax/v200)**2
    #using a polynomial fit of the solution to quickly estimate the initial guess
    p=np.poly1d([ 3.02509213e-03, -1.20296527e-01, 2.34770468e+00, 2.07672808e+01, -2.43183200e+01, 5.44330487e+00])
    init_guess = p(vmax/v200)
    def f_klypin(x):
        return 0.216*x/(np.log(1+x)-x/(1+x))-vmaxperv200sqr
    c = fsolve(f_klypin,init_guess)[0] # numerically solving the transcendental equation above
    Rs = R200/c
    return c, Rs

def get_bullock_spin(jvir,mvir,rvir):
    """
    Function for calculating Bullock spin parameter (https://ui.adsabs.harvard.edu/abs/2001ApJ...555..240B/abstract)
    Expected input:
        - jvir: length of the total angular momentum vector within a virilized sphere in  Msun * Mpc * km/s
        - mvir: virial mass in Msol
        - rvir: virial radius in physical Mpc
    """
    vvir = np.sqrt(G*mvir/rvir) #circular velocity at the virial radius [Vvir^2 = G*Mvir/Rvir] (physical km/s)
    S_Bullock = jvir/ (np.sqrt(2) * mvir * rvir * vvir)
    return S_Bullock

def NFW_profile(r,rho0,Rs):
    """
    A function for calculating the Navarro-Frenk-White profile
    Input:
        - r: distance from the center
        - rho0: density parameter of the NFW profile
        - Rs: scale lenght of the halo
    Returns:
        - NFW profile values at "r" input distances
    """
    rpRs = r/Rs
    return rho0/(rpRs*np.power((1.0+rpRs),2))

def Hz(z, H0, Om, Ol, DE_model, DE_params):
    """
    Hubble parameter at given redshift using astropy.
    Input:
        - z: redshift
        - H0: Hubble constant in km/s/Mpc units
        - Om: non-relativistic matter density parameter
        - Ol: dark energy density parameter
        - DE_model: Dark Energy model name. must be "Lambda", "w0", or "CPL"
        - DE_params: list of the parameters of the DE model.
            -> "Lambda": not used
            -> "w0": [w0]
            -> "CPL": [w0,wa]
    Returns:
        - Hz: Hubble parameter at z redshift in km/s/Mpc units
    """
    if DE_model == "Lambda":
        cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=Ol)
    elif DE_model == 'w0':
        cosmo = wCDM(H0=H0, Om0=Om, Ode0=Ol,w0=DE_params[0])
    elif DE_model == 'CPL':
        cosmo = w0waCDM(H0=H0, Om0=Om, Ode0=Ol,w0=DE_params[0],wa=DE_params[1])
    else:
        raise Exception("Error: unkown dark energy parametrization!\nExiting.\n")
    return cosmo.H(z).value

def get_Delta_c(z, H0, Om, Ol, DE_model, DE_params, mode="BryanEtAl"):
    """
    Virial overdensity constant calculation, see eq 6. of Bryan et al. 1998 ( https://ui.adsabs.harvard.edu/abs/1998ApJ...495...80B/abstract)
    (Assuming Omega_r=0)
    """
    if DE_model == "Lambda":
        cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=Ol)
    elif DE_model == 'w0':
        cosmo = wCDM(H0=H0, Om0=Om, Ode0=Ol,w0=DE_params[0])
    elif DE_model == 'CPL':
        cosmo = w0waCDM(H0=H0, Om0=Om, Ode0=Ol,w0=DE_params[0],wa=DE_params[1])
    else:
        raise Exception("Error: unkown dark energy parametrization!\nExiting.\n")
    x = cosmo.Om0 * (1+z)**3 /(cosmo.H(z).value/H0)**2-1.0
    if mode=="Rockstar":
        # The virial overdensity definition as it is implemented in Rockstar
        return (18.0*np.pi**2 + 82.0*x - 39.0*x**2)/(1.0+x) * cosmo.Om0
    elif mode=="BryanEtAl":
        # The original Bryan & Norman (1998) virial overdensity definition
        return 18.0*np.pi**2 + 82.0*x - 39.0*x**2
    else:
        raise Exception("Error: Unknown virial density definition %s!\nExiting.\n" % (mode))

def get_1D_radial_profile(r,M,Nbins,background_density=0.0):
    """
    This function reconstructs the 1D density profile of a halo
    by using equal "Npart" radial binning
    ---------------------------
    input:
            -r: distances from center
            -M: particle masses
            -Nbins: Number or radial bins
            -method: binning method. Must be "NGP" or "CIC"
    """
    rmax = r[-1]
    NpartTot = len(r)
    NpartPerBin = int(np.floor(NpartTot/Nbins))
    #Allocating memory for the profile
    r_bin_limits = np.zeros(Nbins,dtype=np.double)
    r_bin_centers = np.zeros(Nbins,dtype=np.double)
    rho_bins = np.zeros(Nbins,dtype=np.double)
    # i=0 bin:
    r_bin_limits[0] = r[NpartPerBin-1] #bin upper limit
    r_bin_centers[0] = 0.5*(r_bin_limits[0]) #bin center
    rho_bins[0] = np.sum(M[:NpartPerBin])/(4.0*np.pi/3.0*r_bin_limits[0]**3)
    for i in range(1,Nbins):
        r_bin_limits[i] = r[NpartPerBin*(i+1)-1] #bin upper limit
        r_bin_centers[i] = (0.5*(r_bin_limits[i] + r_bin_limits[i-1])) #bin center
        rho_bins[i] = np.sum(M[NpartPerBin*i:NpartPerBin*(i+1)])/(4.0*np.pi/3.0*(r_bin_limits[i]**3 - r_bin_limits[i-1]**3))
    out_idx = rho_bins > 0.0 # selecting non-empty bins
    rho_bins[out_idx] -= background_density #removing background density
    return r_bin_centers[out_idx], rho_bins[out_idx]


def calculate_halo_shape(coordinates, masses, center_of_mass):
    """
    Calculates halo shape parameters from particle data.

    This function computes the mass distribution tensor, finds its eigen-system,
    and uses it to derive the halo's axis lengths, axis ratios, triaxiality,
    sphericity, and prolateness.

    Inputs:
        *coordinates (np.ndarray): A NumPy array of shape (N, 3) containing the x, y, z coordinates of N particles.
        *masses (np.ndarray): A NumPy array of shape (N,) containing the mass of each particle.
        *center_of_mass (np.ndarray): A NumPy array of shape (3,) for the halo's center of mass [x_cm, y_cm, z_cm].

    Returns:
        *dict: A dictionary containing all the calculated shape parameters.
              Returns None if inputs are invalid (e.g., too few particles).
    """
    if coordinates.shape[0] < 3:
        print("Warning: Not enough particles to define a 3D shape.")
        return None

    # Shift coordinates to be relative to the center of mass
    relative_coords = coordinates - center_of_mass

    # Calculate the components of the mass distribution tensor
    #    I_ij = sum(m_k * x_i * x_j) for all particles k
    I_xx = np.sum(masses * relative_coords[:, 0]**2)
    I_yy = np.sum(masses * relative_coords[:, 1]**2)
    I_zz = np.sum(masses * relative_coords[:, 2]**2)
    I_xy = np.sum(masses * relative_coords[:, 0] * relative_coords[:, 1])
    I_xz = np.sum(masses * relative_coords[:, 0] * relative_coords[:, 2])
    I_yz = np.sum(masses * relative_coords[:, 1] * relative_coords[:, 2])

    mass_distribution_tensor = np.array([
        [I_xx, I_xy, I_xz],
        [I_xy, I_yy, I_yz],
        [I_xz, I_yz, I_zz]
    ])

    # Find the eigenvalues and eigenvectors of the tensor
    #    np.linalg.eigh is used for symmetric matrices like this one.
    eigenvalues, eigenvectors = np.linalg.eigh(mass_distribution_tensor)

    # Sort eigenvalues and eigenvectors in descending order (for a >= b >= c)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # The principal axis lengths are proportional to the square root of the eigenvalues
    a, b, c = np.sqrt(sorted_eigenvalues)

    # Calculating the derived shape parameters
    # Handle the case of a perfect sphere to avoid division by zero
    if np.isclose(a, 0):
        print("Warning: Major axis 'a' is close to zero. Shape is undefined.")
        s = q = triaxiality = sphericity = prolateness = np.nan
    else:
        # Axis Ratios
        s = c / a  # Minor-to-major axis ratio
        q = b / a  # Intermediate-to-major axis ratio

        # Sphericity (often defined as s)
        sphericity = s

        # Triaxiality Parameter T = (a^2 - b^2) / (a^2 - c^2)
        # We can use the eigenvalues (lambda = axis_length^2) directly
        lambda_a, lambda_b, lambda_c = sorted_eigenvalues
        denominator = lambda_a - lambda_c
        if np.isclose(denominator, 0):
             # This is a sphere (a=b=c), so triaxiality is undefined.
            triaxiality = np.nan
        else:
            triaxiality = (lambda_a - lambda_b) / denominator

        # Prolateness p = (a - b) / (a + b)
        denominator_p = a + b
        if np.isclose(denominator_p, 0):
            prolateness = np.nan
        else:
            prolateness = (a - b) / denominator_p


    # Store all results in a single dictionary
    results = {
        'eigenvectors': sorted_eigenvectors,
        'axis_lengths': {'a': a, 'b': b, 'c': c},
        'axis_ratios': {'minor_to_major_s': s, 'intermediate_to_major_q': q},
        'sphericity': sphericity,
        'triaxiality': triaxiality,
        'prolateness': prolateness
    }

    return results