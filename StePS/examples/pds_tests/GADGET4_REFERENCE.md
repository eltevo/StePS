# Building a flat Gadget4 gold-standard reference for PDS validation

The PDS validation harness (`validate_growth.py`) compares a StePS Poincaré-dodecahedral
run to a flat **Gadget4** run built from the **same IC realization**.  Gadget4 is the
reference because it tracks linear growth to ~1%; a StePS-T³ run is *not* a safe reference
(see the caveat at the bottom).  This note records the exact, reproducible setup.

All commands assume the `stepsic` conda env (provides gcc/mpicxx, GSL, HDF5).

## 1. One-time: dependencies and build

Gadget4 needs FFTW3 and zlib in addition to what the env already has:

```bash
conda activate stepsic
conda install -y -c conda-forge fftw zlib          # GSL/HDF5/MPI already present
git clone https://gitlab.mpcdf.mpg.de/vrs/gadget4.git  $REPO/gadget4
```

A `conda` SYSTYPE is provided so Gadget4 finds the env libraries.  In the gadget4 repo:

* `buildsystem/Makefile.path.conda` — points GSL/FFTW/HDF5 at `$CONDA_PREFIX`.
* `Makefile` — add a `SYSTYPE="conda"` block including `Makefile.comp.gcc` +
  `Makefile.path.conda` (next to the `Generic-gcc` block).

**Gotcha:** conda exports a `CPP` env var (the C *preprocessor*) that shadows Gadget4's
`CPP = mpicxx`.  Pass `CPP=mpicxx` on the make command line.  Build into a run dir:

```bash
make -C $REPO/gadget4 DIR=/scratch/.../gadget128_flat SYSTYPE=conda CPP=mpicxx -j8
```

A minimal DM-only `Config.sh` (PERIODIC, SELFGRAVITY, FMM, PMGRID, DOUBLEPRECISION,
DOUBLEPRECISION_FFTW; FOF/SUBFIND/SPH stripped) is enough for a growth reference.

## 2. Per-IC: rescale the IC and write the param

stepsic ICs (`HINDEPENDENT=false`) are in physical Mpc with H0 = 100h km/s/Mpc.  Gadget4
**enforces** the Mpc/h convention (`Hubble=100`) and aborts otherwise, so convert first:

```bash
python rescale_ic_to_gadget.py  testCubic128/.../ic.hdf5  gadget128_flat/ic_mpch.hdf5
```

This multiplies coordinates and BoxSize by `h`, masses by `10h` (1e11 Msun → 1e10 Msun/h),
and leaves velocities unchanged (physical peculiar/√a is h-independent).  Then the param
uses: `Hubble 100`, `HubbleParam h`, `BoxSize 1200*h`, `UnitLength_in_cm 3.085678e24`,
`UnitMass_in_g 1.989e43`, `UnitVelocity_in_cm_per_s 1e5`, softening in Mpc/h, and an
`OutputListFilename` of scale factors `a = 1/(1+z)` matching the StePS run's redshifts.
NTYPES=6 requires Softening{Comoving,MaxPhys}Class0..5 and SofteningClassOfPartType0..5.

## 3. Run

```bash
mpirun -np 62 --bind-to none -x HWLOC_XMLFILE ./Gadget4 flat128.param > run.log 2>&1
```

`--bind-to none` + the conda `HWLOC_XMLFILE` avoid the hwloc topology crash on this node.
2.1M particles to z=0 takes a few minutes on ~62 cores.  Restart-from-checkpoint requires
the *same* rank count, so to change cores restart from the IC (the redo is cheap).

## 4. Validate

```bash
python validate_growth.py --pds <STEPS_PDS_DIR> --gad gadget128_flat \
    --pds-ic <PDS_IC.hdf5> --gad-ic gadget128_flat/ic_mpch.hdf5
```

The harness converts Gadget Mpc/h → physical Mpc, the PDS stereographic coords → physical
geodesic coords, and compares median displacement growth (primary gate), large-scale P(k),
and σ² (resolution-confounded).

## Caveat: do NOT use a StePS-T³ run as the reference

A StePS-T³ run (`testCubic128`, `IS_PERIODIC=2` with a **low-res 63³ Ewald table**) was
found to over-grow ~3× vs Gadget at z=10 from the identical IC — it tracks `D∝a^~1.5`, not
linear.  The PDS mode is immune (it uses the exact 120-image sum, not an Ewald table), but
this is exactly why the flat reference must be Gadget4, not StePS-T³.
