# StePS - STEreographically Projected cosmological Simulations

An N-body code for non-periodic dark matter cosmological simulations

We present a novel N-body simulation method that compactifies the infinite spatial extent of the Universe into a finite sphere with isotropic boundary conditions to follow the evolution of the large-scale structure. Our approach eliminates the need for periodic boundary conditions, a mere numerical convenience which is not supported by observation and which modifies the law of force on large scales in an unrealistic fashion. With this code, it is possible to simulate an infinite universe with unprecedented dynamic range for a given amount of memory, and in contrast of the traditional periodic simulations, its fundamental geometry and topology match observations. We present here the Multi-GPU realization of this algorithm with MPI-OpenMP-CUDA hybrid parallelisation.

This code is under development.

For more information see: [astro-ph](https://arxiv.org/abs/1711.04959)

If you plan to publish an academic paper using this software, please consider citing the following publication:

G. Rácz, I. Szapudi, I. Csabai, L. Dobos, "Compactified Cosmological Simulations of the Infinite Universe": MNRAS, Volume 477, Issue 2, p.1949-1957

![](Images/evolution_full.gif)
![](Images/evolution_center.gif)
![](Images/rotation.gif)

![alt text](Images/LCDM_SP_0500Mpc_64M_StePS_glass_tilefac3_LBOX2214.png "A slice from a simulation volume.")
![alt text](Images/LCDM_SP_0500Mpc_64M_StePS_glass_tilefac3_LBOX738.png "The center of the slice.")

![alt text](Images/VOI100_InnerRegion.png "Particles in the center of a simulation volume.")
![alt text](Images/VOI100_Disp_InnerRegion.png "The displacement field in the same volume at z=0.")
