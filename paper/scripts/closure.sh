#!/bin/bash
# Runs for the bubble-closure subsection (Fig. closure): the pinched-witness case of the paper with
# the box extended to xi = -12 (bubble closure at xi ~ -9), single quasi-static solve.
#   cold / smoothed (plasma.smooth_length = 0.005) on the stretched grid, dxi = 0.005 ... 0.000625;
#   smoothed with axis cells 1e-3 and 2.5e-4; cold on the uniform grids 0.01 and 0.005.
# Run from a scratch directory; QUARZ=/path/to/quarz selects the executable. Then: python3 fig_closure.py
set -e
Q=${QUARZ:-quarz}
IN=$(cd "$(dirname "$0")/../inputs" && pwd)
O="xi.min=-12 output.field_files=0"
run() { local n=$1 deck=$2; shift 2; [ -f out_closure_$n/axis_000000.txt ] && return
        $Q $IN/$deck $O "$@" output.dir=out_closure_$n > log_closure_$n.txt; }
for d in 0.005 0.0025 0.00125 0.000625; do
    run cold_$d conv_s1.in xi.step=$d
    run a5e-3_$d conv_s1.in xi.step=$d plasma.smooth_length=0.005
done
for h in 0.001 0.00025; do
    run a5e-3_h$h conv_s1.in xi.step=0.00125 plasma.smooth_length=0.005 "grid.regions=0.05:$h 2.5:0.01 8.0:0.05"
done
for u in 0.01 0.005; do for d in 0.005 0.00125; do run u${u}_$d conv_u$u.in xi.step=$d; done; done
