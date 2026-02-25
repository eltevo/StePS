# Plans to upgrade StePS to handle Poincare dodecahedral space (PDS) topology


### Rough roadplan

- First, you have to define a new boundary condition (topological manifold) option in the makefile (Makefile, Template-Darwin-Makefile, Template-LinuxGCC-Makefile, Template-LinuxICC-Makefile), and make sure that the compiled code will see it.
- Second, The new boundary condition must be defined in the step.cc: we have to tell the code what happens if a particle leaves the simulation volume.
- After this, a new lookup table construction function must be written in ewald_space.cc that describes the gravitational effect of all periodic images (similarly to the implemented T^3 and S1xR^2 lookup table generation) Note that the discrete symmetry group of the target topology can be used to speed up the table building. The corresponding I/O functions for saving/loading the table must be added to the inputoutput.cc.
- The table construction and I/O calls must be added to the main.cc to be able to generate and save/load the table when the simulation starts. See the existing T^3 and S^1xR^2 for an example.
- The force calculation method must be added to the forces.cc and forces_cuda.cu, to use the pre-calculated lookup table during the force calcumation. The values of the interpolated table must be added to the nearest-image force, if IS_PERIODIC>1. For IS_PERIODIC==1, only the nearest image should be used.
-  The I/O functions in inputoutput.cc should also be updated, since the headers of the output logfiles and snapshots should contain the information about the new topology (parameters of the volume, name of the manifold, etc.)
- The biggest challenge may not be modifying the simulation code, but rather generating the IC. Since we’re not in T³ space, and the space isn’t flat either, we can’t use standard Fourier transforms to generate the initial density field from the cosmological power spectrum. Furthermore, since space is not flat, We cannot use a grid as a particle load either, we may need a special glass.


### Preliminary evaluation and speculations on possible strategies and challenges 

#### What StePS actually does

StePS uses stereographic projection to map the infinite three-dimensional hyperplane of space onto the surface of a compact four-dimensional sphere. Initial conditions are defined with the help of an equal-volume grid on this compact surface.

The StePS algorithm eliminates the need for periodic boundary conditions and can simulate an infinite Universe with a topology that matches observations. The price is that most of the simulated volume has smoothly varying mass and spatial resolution, carrying different systematics than periodic simulations.

StePS provides a direct O(N²) multi-GPU-accelerated gravity solver for maximal accuracy and a Barnes-Hut octree O(N log N) solver for large CPU clusters.

The crucial clarification: StePS uses S³ as a **coordinate compactification trick** for a flat, infinite universe. It is not simulating a universe whose physical geometry IS S³. The gravitational forces remain Newtonian 1/r², computed using Euclidean distances after stereographic projection. The varying resolution comes from the Jacobian of the projection.

---

#### Where StePS genuinely helps for PDS

The overlap is real and meaningful in several areas:

**Coordinate system already lives on S³.** This is the single biggest practical benefit. StePS already represents particles as points on S³ embedded in R⁴ (unit quaternions or 4D unit vectors). All the coordinate infrastructure — updating positions on the sphere, computing angular separations, the equal-volume grid — is already built. For PDS you need exactly this. Starting from PySCo or monofonIC, you'd spend months building this; with StePS it's already there.

**Direct summation GPU kernel is the right architecture.** For PDS with the method-of-images gravity (summing over 120 I* image copies of each particle), an O(N²) direct summation solver is exactly what you need — no FFT, no mesh, just pairwise forces. The StePS approach can achieve unprecedented dynamic range by using a small number of particles, and the relatively small number of particles makes the use of direct force summation possible with low memory needs. The approach is ideal for a relatively simple and very effective GPU parallelization. Modifying the force kernel to sum over 120 image copies rather than one copy is a contained change to existing, validated GPU code.

**No FFT dependency at all.** Unlike every PM-based code, StePS has no FFTW, no mesh Poisson solver, no periodic assumptions buried in the gravity module. This is what you want — the whole point of PDS is that you can't use FFT.

**Recently extended and actively maintained.** The cylindrical StePS extension introduces a compactified simulation framework periodic along a single axis with infinite topology in the perpendicular directions, demonstrating the framework's flexibility for non-standard geometries. The authors have shown willingness to implement non-trivial topologies, and are from ELTE in Budapest — which is potentially directly relevant to you.

**IC generation is grid-based on S³.** Initial conditions are defined with the help of an equal-volume grid on the compact S³ surface. This is closer conceptually to what PDS ICs need than a flat cubic Fourier-mode grid.

---

#### Where StePS fundamentally does NOT close the gap

Here is where you need to be clear-eyed about what remains:

**The gravity physics is wrong for PDS — and this is the hardest problem.** StePS computes 1/r² Newtonian gravity between particles after stereographic mapping. It is simulating flat-space physics in spherical coordinates. For genuine PDS, gravity must use the curved-space Green's function on S³:

G(χ) = (π − χ) / (4π² R sin χ)

where χ is the geodesic separation. This differs significantly from 1/r at scales comparable to the curvature radius R. Changing the force kernel formula is simple; validating that the new force law is physically correct and numerically stable is not.

**No face identifications / holonomy group.** StePS has no boundary identifications at all — it's an open sphere with isotropic outer boundary conditions. PDS requires 12 specific face-pair identifications, each involving a π/5 rotation from the binary icosahedral group I*. These must be added from scratch.

**Initial conditions still use flat-space Fourier modes.** StePS adapts standard Zeldovich/LPT ICs to its grid, but still draws amplitudes from a flat-space power spectrum with continuous k. PDS requires discrete I*-invariant eigenmodes of S³ — the power on large scales is fundamentally suppressed, and the allowed modes are a specific discrete set. This is the IC generation problem described before; StePS does not solve it.

**Flat cosmological background.** StePS assumes Ω_total = 1. PDS needs a curved Friedmann background with Ω_total ≈ 1.018, which changes growth factors and the expansion history.



