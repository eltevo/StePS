#define pi 3.14159265358979323846264338327950288419716939937510
#define UNIT_T 47.14829951063323 //Unit time in Gy
#define UNIT_V 20.738652969925447 //Unit velocity in km/s

extern double L; //Size of the simulation box
extern double SPHERE_DIAMETER; //diameter of the 4D sphere in Mpc used in the stereographic projection
extern double R_CUT;
extern int N_SIDE, R_GRID, N_glass;
extern char IC_FILE[1024]; //input file
extern char SphericalGlassFILE[512]; //input spherical glass file
extern char OUT_FILE[1024]; //output directory
extern int IC_FORMAT; // 0: ascii, 1:GADGET
extern int SPHERICAL_GLASS; // 0: using Healpix for IC generation 1: using spherical glassfile of IC generation
extern int FOR_COMOVING_INTEGRATION;
extern int RANDOM_SEED; //random seed for the random rotation of the shells
extern int RANDOM_ROTATION; //if 1 using random rotation, otherwise the code will not rotate randomly the HEALPix shells
extern int NUMBER_OF_INPUT_FILES; //Number of files that are written in parallel by the periodic IC generator
extern int TILEFAC;

extern double a_max; //maximal scalefactor

extern double* M; //Particle masses
extern double M_tmp;
extern unsigned long long int N; //Number of particles if we use one input file, or the number of particles in the actual input file
extern unsigned long long int N_IC_tot; //total Number of particles in the input files
extern unsigned long int N_out; //Number of output particles
extern double** x; //particle coordinates and velocities
extern double** SphericalGlass; //xyz coordinates of the sperical glass
extern double** x_out; //Output particle coordinates and velocities
extern long int* COUNT;

extern double G; //Newtonian gravitational constant

//Cosmological parameters
extern double Omega_b,Omega_lambda,Omega_dm,Omega_r,Omega_k,Omega_m,H0,H0_start,Hubble_param;
extern double rho_crit; //Critical density
extern double a, a_start; //Scalefactor, scalefactor at the starting time, previous scalefactor

extern double dist_unit_in_kpc;
extern double VOI[3]; //The (x,y,z) coordinates of the center of volume of interest(VOI)

//kdtree variables
extern kdtree* tree;
extern int *index_list;
