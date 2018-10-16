#define pi 3.14159265358979323846264338327950288419716939937510
#define UNIT_T 47.14829951063323 //Unit time in Gy
#define UNIT_V 20.738652969925447 //Unit velocity in km/s

#ifdef GLASS_MAKING //Newtonian gravitational constant (in internal units)
#define G -1.0 //Gravity is repulsive, if Glassmaking is on.
#else
#define G 1.0 //Normal gravity
#endif

#ifdef USE_SINGLE_PRECISION
typedef float REAL;
#else
typedef double REAL;
#endif

extern int IS_PERIODIC; //periodic boundary conditions, 0=none, 1=nearest images, 2=ewald forces
extern int COSMOLOGY; //Cosmological Simulation, 0=no, 1=yes
extern int COMOVING_INTEGRATION; //Comoving integration 0=no, 1=yes, used only when  COSMOLOGY=1
extern REAL L; //Size of the simulation box
extern char IC_FILE[1024]; //input file
extern char OUT_DIR[1024]; //output directory
extern char OUT_LST[1024]; //output redshift list file. only used when OUTPUT_TIME_VARIABLE=1
extern int IC_FORMAT; // 0: ascii, 1:GADGET
extern int OUTPUT_FORMAT; //Output format 0: ASCII 2:HDF5
extern int OUTPUT_TIME_VARIABLE; // 0: time, 1: redshift
extern double MIN_REDSHIFT; //The minimal output redshift. Lower redshifts considered 0.
extern int REDSHIFT_CONE; // 0: standard output files 1: one output redshift cone file
extern int HAVE_OUT_LIST;
extern double *out_list; //Output redshits
extern double *r_bin_limits; //bin limints in Dc for redshift cone simulations
extern int out_list_size; //Number of output redshits
extern bool* IN_CONE;

extern int n_GPU; //number of cuda capable GPUs
//variables for MPI
extern int numtasks, rank;
extern int N_mpi_thread; //Number of calculated forces in one MPI thread
extern int ID_MPI_min, ID_MPI_max; //max and min ID of of calculated forces in one MPI thread
extern MPI_Status Stat;
extern REAL* F_buffer; //buffer for force copy
extern int BUFFER_start_ID;

extern int e[2202][4]; //ewald space
extern int H[2202][4]; //ewald space

extern REAL x4, err, errmax, ACC_PARAM; //variables used for error calculations
extern double h, h_min, h_max,  t_next; //actual stepsize, minimal and maximal stepsize, next time for output
extern double a_max,t_bigbang; //maximal scalefactor; Age of Big Bang

extern double FIRST_T_OUT, H_OUT; //First output time, output frequency in Gy

extern REAL* M; //Particle masses
extern REAL M_tmp;
extern int N; //Number of particles
extern int t; //Number of the actual timestep
extern REAL* x; //particle coordinates
extern REAL* v; //and velocities
extern REAL* F; //Forces
extern REAL w[3]; //Parameters for smoothing in force calculation
extern REAL beta; //Particle radii
extern REAL ParticleRadi; //Particle radii; readed from parameter file
extern REAL rho_part; //One particle density
//extern REAL SOFT_CONST[8]; //Parameters for smoothing in force calculation
extern REAL M_min; //minimal particle mass
extern REAL mass_in_unit_sphere; //Mass in unit sphere

//Cosmological parameters
extern double Omega_b,Omega_lambda,Omega_dm,Omega_r,Omega_k,Omega_m,H0,Hubble_param, Decel_param, delta_Hubble_param, Hubble_tmp;
extern double rho_crit; //Critical density
extern double a, a_start, a_prev, a_tmp; //Scalefactor, scalefactor at the starting time, previous scalefactor
extern double T, delta_a, Omega_m_eff; //Physical time, change of scalefactor, effectve Omega_m

//Functions
//Initial timestep length calculation
double calculate_init_h();
//Functions used for the Friedmann-equation
double friedman_solver_step(double a0, double h, double Omega_lambda, double Omega_r, double Omega_m, double Omega_k, double H0);
void recalculate_softening();
//This function calculates the deceleration parameter
double CALCULATE_decel_param(double a);
