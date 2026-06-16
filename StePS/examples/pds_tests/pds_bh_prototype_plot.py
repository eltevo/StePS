#!/usr/bin/env python3
"""Run the PDS Barnes-Hut prototype on a few snapshots and plot the
accuracy/speed tradeoff vs the opening angle theta.

Usage:
    ./pds_bh_prototype_plot.py <snap1.hdf5> [snap2.hdf5 ...] [--radii R] [--nsample N]

Produces pds_bh_tradeoff.png.  Requires the compiled ./pds_bh_prototype binary.
"""
import re, subprocess, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

BIN = str(Path(__file__).with_name("pds_bh_prototype"))

ROW = re.compile(r"^\s*([0-9.]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)"
                 r"\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.]+)x")

def run(snap, radii, nsample):
    out = subprocess.run([BIN, snap, str(radii), str(nsample)],
                         capture_output=True, text=True).stdout
    theta, mean, p99, speed = [], [], [], []
    for line in out.splitlines():
        m = ROW.match(line)
        if m:
            theta.append(float(m.group(1)))
            mean.append(float(m.group(2)))
            p99.append(float(m.group(4)))
            speed.append(float(m.group(8)))
    return map(np.array, (theta, mean, p99, speed))

def zlabel(snap):
    with h5py.File(snap, "r") as f:
        return float(f["Header"].attrs["Redshift"])

def main():
    args = [a for a in sys.argv[1:]]
    radii, nsample = 10.0, 1500
    if "--radii" in args:
        radii = float(args[args.index("--radii") + 1])
    if "--nsample" in args:
        nsample = int(args[args.index("--nsample") + 1])
    snaps = [a for a in args if a.endswith(".hdf5")]
    if not snaps:
        print("no snapshots given"); sys.exit(2)

    fig, (axE, axS) = plt.subplots(1, 2, figsize=(13, 5.2))
    for snap in snaps:
        z = zlabel(snap)
        theta, mean, p99, speed = run(snap, radii, nsample)
        lbl = f"z = {z:.2f}"
        l, = axE.plot(theta, 100*mean, "o-", label=lbl + "  (mean)")
        axE.plot(theta, 100*p99, "x--", color=l.get_color(), alpha=0.6,
                 label=lbl + "  (99th pct)")
        axS.plot(theta, speed, "o-", color=l.get_color(), label=lbl)

    for ax in (axE, axS):
        ax.set_xlabel(r"opening angle  $\theta$")
        ax.grid(alpha=0.3, which="both")
        ax.invert_xaxis()  # more accurate to the right
    axE.set_yscale("log"); axE.set_ylabel("relative force error  [%]")
    axE.set_title("PDS Barnes-Hut accuracy")
    axE.axhspan(0.1, 1.0, color="green", alpha=0.08)
    axE.text(axE.get_xlim()[0], 0.32, " 0.1-1% target band", color="green",
             fontsize=8, va="center")
    axE.legend(fontsize=8)
    axS.set_yscale("log"); axS.set_ylabel(f"speedup vs exact  (N={_N(snaps[0])})")
    axS.set_title("PDS Barnes-Hut speedup")
    axS.legend(fontsize=8)
    fig.suptitle("PDS (S$^3$/I*) Barnes-Hut prototype: accuracy / speed vs opening angle",
                 fontweight="bold")
    fig.tight_layout()
    out = Path(__file__).with_name("pds_bh_tradeoff.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

def _N(snap):
    with h5py.File(snap, "r") as f:
        return int(f["Header"].attrs["NumPart_ThisFile"][1])

if __name__ == "__main__":
    main()
