#!/bin/sh
# The note on coarse radial cells inside the plasma of a long box (Simulation.cpp): printed once for a box longer
# than 300 c/omega_p whose loaded plasma rings sit in cells above 0.03, and not otherwise.
#   sh tests/long_box_note.sh <quarz executable>
Q=$1
SRC=$(cd "$(dirname "$0")/.." && pwd)
W=$(mktemp -d "${TMPDIR:-/tmp}/quarz_note.XXXXXX")
trap 'rm -rf "$W"' EXIT
cd "$W" || exit 1
fail=0
cat > base.in <<'IN'
grid.type = regions
grid.rmax = 10
xi.min = 0
xi.step = 0.05
time.steps = 0
plasma.species = electrons
plasma.rmax = 7
electrons.ppc = 2
beams = driver
driver.rigid = 1
driver.n0 = 0.01
driver.sigma_r = 1
driver.sigma_xi = 2
driver.xi0 = 8
driver.gamma = 1000
driver.nparticles = 20000
output.every = 0
IN
run() {   # name, expected number of notes, overrides
  name=$1; want=$2; shift 2
  "$Q" base.in output.dir=out_$name "$@" > $name.log 2>&1 || { echo "FAILED: $name: run failed"; tail -3 $name.log; fail=1; return; }
  got=$(grep -c '^Note: the box is' $name.log)
  if [ "$got" = "$want" ]; then echo "  $name: $got note(s), as expected"
  else echo "FAILED: $name: $got note(s), expected $want"; fail=1; fi
}
run long_coarse   1 xi.max=320 "grid.regions=2.0:0.01 10.0:0.1"
run long_fine     0 xi.max=320 "grid.regions=2.0:0.01 7.3:0.02 10.0:0.1"
run short_coarse  0 xi.max=40  "grid.regions=2.0:0.01 10.0:0.1"
run long_coarse_outside 0 xi.max=320 "grid.regions=2.0:0.01 10.0:0.1" plasma.rmax=2
run long_steps    1 xi.max=320 "grid.regions=2.0:0.01 10.0:0.1" time.steps=2 time.dt=10
grep -q 'reach h = 0.1' long_coarse.log || { echo "FAILED: the note does not name the cell size 0.1"; grep '^Note' long_coarse.log; fail=1; }
[ $fail = 0 ] && echo "long box note: ok" || echo "long box note test FAILED"
exit $fail
