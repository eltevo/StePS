#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "Usage: sh $0 <StePS executable> <r3|cylinder|pds|t3>" >&2
    exit 2
fi
binary=$(realpath "$1")
geometry=$2
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
x1=-1
x2=1
case "$geometry" in
    r3)
        periodic=0
        mass=13.671
        z1=-1
        z2=1 ;;
    pds)
        periodic=1
        mass=13.671
        z1=-1
        z2=1 ;;
    t3)
        periodic=1
        mass=51
        x1=2
        x2=8
        z1=2
        z2=8 ;;
    cylinder)
        periodic=1
        mass=25.63
        z1=1
        z2=8 ;;
    *) echo "Expected r3, cylinder, pds or t3" >&2; exit 2 ;;
esac
cat > "$tmp/ic.txt" <<EOF
$x1 $x1 $z1 0 0 0 $mass
$x2 $x1 $z1 0 0 0 $mass
$x1 $x2 $z1 0 0 0 $mass
$x2 $x2 $z1 0 0 0 $mass
$x1 $x1 $z2 0 0 0 $mass
$x2 $x1 $z2 0 0 0 $mass
$x1 $x2 $z2 0 0 0 $mass
$x2 $x2 $z2 0 0 0 $mass
EOF
cp "$tmp/ic.txt" "$tmp/glass.txt"
cat > "$tmp/test.param" <<EOF
Omega_b 0.0
Omega_lambda 0.7
Omega_m 0.3
Omega_r 0.0
HubbleConstant 70.0
a_start 0.99
a_max 0.990001
COSMOLOGY 1
IS_PERIODIC $periodic
COMOVING_INTEGRATION 1
L_BOX 10
R_SIM 4
PDS_R_CURV 8
IC_FILE $tmp/ic.txt
IC_FORMAT 0
OUT_DIR $tmp/
OUTPUT_TIME_VARIABLE 1
OUTPUT_FORMAT 0
REDSHIFT_CONE 0
MIN_REDSHIFT 0
ACC_PARAM 0.01
RADIAL_FORCE_ACCURACY 40
RADIAL_FORCE_TABLE_SIZE 8
STEP_MIN 0.001
STEP_MAX 0.001
PARTICLE_RADII 0.1
FIRST_T_OUT 1
H_OUT 1
RADIAL_BH_FORCE_CORRECTION 1
GLASS_FILE_FOR_BH_FORCE_CORRECTION $tmp/glass.txt
RADIAL_BH_FORCE_TABLE_SIZE 8
RADIAL_BH_FORCE_TABLE_ITERATION 2
EOF

run() {
    (cd "$tmp" && OMP_NUM_THREADS=1 mpirun -np 2 "$binary" "$tmp/test.param" 1) > "$tmp/log" 2>&1 || {
        cat "$tmp/log" >&2
        exit 1
    }
}
if [ "$geometry" = pds ] || [ "$geometry" = t3 ]; then
    table="$tmp/radial_bh_force_table.dat"
    printf 'untouched PDS table\n' > "$table"
    cp "$table" "$tmp/expected"
    rm "$tmp/glass.txt"
    run
    grep -q 'radial BH force correction is not used' "$tmp/log"
    if [ "$geometry" = pds ]; then
        grep -q 'PDS Barnes-Hut force calculation finished' "$tmp/log"
    fi
    cmp "$table" "$tmp/expected"
    cp "$tmp/z1000.dat" "$tmp/expected-snapshot"
    rm "$table"
    run
    test ! -e "$table"
    cmp "$tmp/z1000.dat" "$tmp/expected-snapshot"
    sed 's/^RADIAL_BH_FORCE_CORRECTION 1$/RADIAL_BH_FORCE_CORRECTION 0/' "$tmp/test.param" > "$tmp/off.param"
    cp "$tmp/off.param" "$tmp/test.param"
    run
    test ! -e "$table"
    cmp "$tmp/z1000.dat" "$tmp/expected-snapshot"
    echo "$geometry ignores radial BH correction."
    exit 0
fi
expect_generated() {
    grep -q 'Radial BH force correction table calculated' "$tmp/log"
    if grep -q 'Loaded radial BH force correction table' "$tmp/log"; then
        echo "Unexpected cache hit" >&2
        exit 1
    fi
}
expect_loaded() {
    grep -q 'Loaded radial BH force correction table' "$tmp/log"
    if grep -q 'Radial BH force correction table calculated' "$tmp/log"; then
        echo "Unexpected table regeneration" >&2
        exit 1
    fi
}

run
expect_generated
table="$tmp/radial_bh_force_table.dat"
grep -q "^# Radius 4$" "$table"
grep -q "^# Theta " "$table"
cp "$table" "$tmp/expected"
cp "$tmp/z1000.dat" "$tmp/expected-snapshot"
sed -n '/Initial force calculation.../,$p' "$tmp/log" |
    grep '^MPI task 0: Octree force calculation started with random' > "$tmp/expected-random" || :
rm "$tmp/glass.txt"
run
expect_loaded
cmp "$table" "$tmp/expected"
cmp "$tmp/z1000.dat" "$tmp/expected-snapshot"
sed -n '/Initial force calculation.../,$p' "$tmp/log" |
    grep '^MPI task 0: Octree force calculation started with random' > "$tmp/actual-random" || :
cmp "$tmp/expected-random" "$tmp/actual-random"
cp "$tmp/ic.txt" "$tmp/glass.txt"

for field in Geometry Radius Theta Bins ParticleRadius Cosmology Randomization Particles; do
    sed "s/^# $field .*/# $field invalid/" "$tmp/expected" > "$table"
    run
    expect_generated
    cmp "$table" "$tmp/expected"
done
if [ "$geometry" = cylinder ]; then
    sed 's/^# Cylinder .*/# Cylinder invalid/' "$tmp/expected" > "$table"
    run
    expect_generated
    cmp "$table" "$tmp/expected"
fi
sed 's/^# Particles .*/# Particles 9/' "$tmp/expected" > "$table"
run
expect_generated
grep -q 'particle count.*differs' "$tmp/log"
cmp "$table" "$tmp/expected"

sed '$d' "$tmp/expected" > "$table"
run
expect_generated
cmp "$table" "$tmp/expected"
sed '$s/[^[:space:]]*$/nan/' "$tmp/expected" > "$table"
run
expect_generated
cmp "$table" "$tmp/expected"
printf '# Radial BH force correction table\n# r[Mpc] F_correction\n' > "$table"
run
expect_generated
cmp "$table" "$tmp/expected"

if [ "$geometry" = cylinder ] && grep -q '^# PeriodicForce real-space$' "$table"; then
    cp "$tmp/test.param" "$tmp/cosmological.param"
    sed -e 's/^COSMOLOGY .*/COSMOLOGY 0/' \
        -e 's/^COMOVING_INTEGRATION .*/COMOVING_INTEGRATION 0/' \
        -e 's/^OUTPUT_TIME_VARIABLE .*/OUTPUT_TIME_VARIABLE 0/' \
        -e 's/^a_max .*/a_max 0.00001/' \
        -e 's/^H_OUT .*/H_OUT 0.000001/' \
        -e 's/^IS_PERIODIC .*/IS_PERIODIC 2/' \
        "$tmp/cosmological.param" > "$tmp/test.param"
    run
    expect_generated
    set -- "$tmp"/t*.dat
    cp "$1" "$tmp/noncosmological-snapshot"
    sed -n '/Initial force calculation.../,$p' "$tmp/log" |
        grep '^MPI task 0: Octree force calculation started with random' > "$tmp/expected-random" || :
    run
    expect_loaded
    cmp "$1" "$tmp/noncosmological-snapshot"
    sed -n '/Initial force calculation.../,$p' "$tmp/log" |
        grep '^MPI task 0: Octree force calculation started with random' > "$tmp/actual-random" || :
    cmp "$tmp/expected-random" "$tmp/actual-random"
    cp "$tmp/cosmological.param" "$tmp/test.param"
    cp "$tmp/expected" "$table"
fi

printf '0.5 0.5 0.5 0 0 0 %s\n' "$mass" >> "$tmp/ic.txt"
if (cd "$tmp" && OMP_NUM_THREADS=1 timeout 30 mpirun -np 2 "$binary" "$tmp/test.param" 1) > "$tmp/log" 2>&1; then
    echo "Expected a glass/IC particle count mismatch to fail" >&2
    exit 1
fi
grep -q 'the number of particles in the radial BH force correction (8) does not match the number of particles in the initial conditions (9)' "$tmp/log"
echo "Radial BH cache tests passed ($geometry)."
