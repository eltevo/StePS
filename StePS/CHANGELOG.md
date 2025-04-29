# Change Log
All notable changes to the StePS simulation code will be documented in this file.

## [TBD] - TBD

### Added
- Cylindrically symmetric boundary conditions

### Changed
- Updated makefile templates

### Fixed
- Fixed constant-resolution periodic initial condition reading from HDF5 format.
- Fixed H0 independent unit bugs

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
