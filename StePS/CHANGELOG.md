# Change Log
All notable changes to the StePS simulation code is documented in this file.

## [v2.0.0.0] - 2026-02-20

### Added
- Barnes-Hut (Octree) force calculation option (CPU only) [J. Barnes, P. Hut Nature 324 (6096) (1986) 446–449.]
- Random domain center shift option for periodic (T^3) Barnes-Hut simulations
- Random domain center shift and rotation option for spherical (R^3) Barnes-Hut simulations
- Random domain center shift and rotation in cylindrical (S^1xR^2) Barnes-Hut simulations
- Implemented S^1xR^2 topological manifold (cylindrically symmetric boundary conditions)
- Implemented Ewald summation in S^1xR^2 topology [Tornberg, A.-K. 2015, Advances in Computational Mathematics, 42, 227–248]
- Added radial force correction option in R^3 and S^1xR^2 Octree methods for radial simulation stability 
- Ewald lookup table I/O (for both T^3 and S^1xR^2 topological manifolds)
- Glass making logfile is produced during glass making

### Changed
- Updated makefile templates
- w0 and wa parameter values of wCDM and w0waCDM cosmologies are saved into the hdf5 snapshots
- Simulation radius is saved into the hdf5 snapshot header
- Simulation geometry (T^3, R^3, or S^1xR^2) is saved into the hdf5 snapshot header
- Cosmological parameters are saved to the logfiles
- Executable info (version, git commit ID, git branch, compiler, build date) are saved into the hdf5 snapshots and logfiles.
- Optimized gravitational softening calculation on GPUs
- Optimized force calculation CUDA kernels
- Individual GPU force calculation time is printed out
- Number of OpenMP and MPI threads are printed out during startup.

### Fixed
- Fixed constant-resolution periodic initial condition reading from HDF5 format
- Fixed H0 independent unit bugs
- Fixed high-accuracy Ewald summation option in fully periodic (T^3) simulations
- Simulation box size is saved properly in 32bit mode.
- Fixed tstart and ASCII format overwrite bugs.

## [v1.0.2.2] - 2024-10-25

### Added
- Added error messages for non-comoving cosmological simulations
- Added better descriptions to the README file

### Changed
- Updated linux-gcc makefile template

### Fixed
- Fixed malloc bug in the read_OUT_LST function
- Fixed memory allocation typos in the HDF5 reader function

## [v1.0.2.0] - 2024-01-23


### Added
- Added error message for non-comoving non-standard simulations
- Added non-comoving example simulation

### Changed
- Omega_dm parameter is changed to Omega_m in the paramfile.
- Updated example simulations (better filenames, Planck 2018 parameters, updated readme)
- Simulation wall-clock time is written out in hours too at the end of the simulation.
- Next output time/redshift is written to stdout in every timestep.

### Fixed
- Fixed redshift output bug

## [v1.0.1.0] - 2022-07-11


### Added
- Added option for wCDM cosmology parametrization
- Added option for w0waCDM cosmology parametrization
- Added option for using tabulated expansion history

### Changed

### Fixed
- Fixed deceleration parameter calculation


## [v1.0.0.0] - 2022-02-28

First github release.

### Main features
- Dark matter only LambdaCDM cosmological N-body simulations
- Parallelized with MPI, OpenMP and CUDA
- Direct force calculation
- HDF5, Gadget2 and ASCII input formats
- ASCII an HDF5 output formats
- Options for standard periodic and non-periodic spherical cosmological simulations
- Periodic, quasi-periodic or spherical glass generation
