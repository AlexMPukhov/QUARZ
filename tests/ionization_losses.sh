#!/bin/sh
# Ionization losses of the laser (laser.ionization_losses): energy balance against the electrons born in the run
# (validation/ion_loss.in, ion_loss_check.py; README section 12). Short version: 20 steps, gas.ppc = 4.
#   sh tests/ionization_losses.sh <quarz executable>
Q=$1
SRC=$(cd "$(dirname "$0")/.." && pwd)
W=$(mktemp -d "${TMPDIR:-/tmp}/quarz_ionloss.XXXXXX")
trap 'rm -rf "$W"' EXIT
cd "$W" || exit 1
D=$SRC/validation/ion_loss.in
S="time.steps=20 gas.ppc=4"
"$Q" $D $S output.dir=on > on.log 2>&1 || { echo "FAILED: run (losses on)"; tail -5 on.log; exit 1; }
"$Q" $D $S output.dir=off laser.ionization_losses=0 > off.log 2>&1 || { echo "FAILED: run (losses off)"; tail -5 off.log; exit 1; }
"$Q" $D $S output.dir=pl plasma.species=electrons electrons.density=0.3 electrons.ppc=4 > pl.log 2>&1 ||
    { echo "FAILED: run (pre-ionized)"; tail -5 pl.log; exit 1; }
"$Q" $D $S output.dir=noip laser.ionization_potential_loss=0 > noip.log 2>&1 ||
    { echo "FAILED: run (losses without Ip)"; tail -5 noip.log; exit 1; }
grep -q 'ionization losses: on (' on.log || { echo "FAILED: the run summary does not report the losses"; exit 1; }
grep -q 'ionization losses: on without Ip' noip.log || { echo "FAILED: the run summary does not report 'without Ip'"; exit 1; }
python3 "$SRC/validation/ion_loss_check.py" on off pl 13.598 0.03 noip
