#!/bin/bash
# Validation suite of QUARZ (results quoted in README §7).
#
# usage: cd validation && ./run_validation.sh [options] [sections]
#   sections   numbers 1..25 (default: all)
#   -q         quick mode: reduced sizes, ~2-3 min on 2 cores (regression test after code changes)
#   -j N       run N sections in parallel (OpenMP threads are split between them)
#   -c         compare with the stored reference (reference_full.txt / reference_quick.txt):
#              PASS/FAIL per section, exit code 1 on failure
#   -x PATH    executable (default ../build/quarz)
#   -u         update the stored reference with this run (after checking the results!)
# needs python3 + numpy + scipy (+ openpmd_api for section 10, mpirun for 9)
set -u
cd "$(dirname "$0")"
QS=../build/quarz; QUICK=0; JOBS=1; CMP=0; UPD=0
while getopts "qj:cx:u" o; do case $o in q) QUICK=1;; j) JOBS=$OPTARG;; c) CMP=1;; x) QS=$OPTARG;; u) UPD=1;; *) exit 2;; esac; done
shift $((OPTIND-1))
SECTIONS=${*:-$(seq 1 25)}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
NCORES=$(nproc); export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( NCORES / JOBS > 0 ? NCORES / JOBS : 1 ))}
MPIRUN=""; if command -v mpirun > /dev/null 2>&1; then MPIRUN="mpirun --oversubscribe -np"; [ "$(id -u)" = "0" ] && MPIRUN="mpirun --allow-run-as-root --oversubscribe -np"; fi
# q FULL QUICK: choose a value by mode
q() { if [ "$QUICK" = 1 ]; then echo "$2"; else echo "$1"; fi; }

sec1() {
    echo "== 1. linear wake vs Green's function (n_b = 0.001) =="
    $QS linear.in driver.n0=0.001 output.dir=out_lin_u > /dev/null
    python3 check_linear.py out_lin_u "uniform dr=0.01         " 0.001 2>/dev/null
    $QS linear.in driver.n0=0.001 output.dir=out_lin_s grid.type=regions "grid.regions=0.3:0.002 2:0.02 8:0.05" \
        grid.max_ratio=1.05 > /dev/null
    python3 check_linear.py out_lin_s "stretched 0.002 -> 0.05 " 0.001 2>/dev/null
}

sec2() {
    echo "== 2. blowout: resolution study =="
    run() { name=$1; shift; $QS blowout.in output.dir=out_b_$name xi.max=10 "$@" > /dev/null; python3 bsum.py out_b_$name $name; }
    run uniform_dr0.01_dxi0.01
    if [ "$QUICK" = 0 ]; then
        run uniform_dr0.005_dxi0.005 grid.dr=0.005 xi.step=0.005
        run uniform_dr0.0025_dxi0.0025 grid.dr=0.0025 xi.step=0.0025
    fi
    run stretched_dxi0.005 xi.step=0.005 grid.type=regions "grid.regions=0.5:0.0025 2.5:0.01 8:0.05" grid.max_ratio=1.05
    run stretched_dxi0.005_PCR xi.step=0.005 grid.type=regions "grid.regions=0.5:0.0025 2.5:0.01 8:0.05" \
        grid.max_ratio=1.05 solver.tridiag=pcr
}

sec3() {
    echo "== 3. betatron oscillation in an ion channel =="
    $QS betatron.in > /dev/null
    python3 -c "
import sys; sys.path.insert(0,'../tools')
from quarz_read import read_beamlog
import numpy as np
b=read_beamlog('out_betatron/beams.txt')['witness']; t=b['t']; rr=b['r_rms']; w=1/np.sqrt(2*1000)
print('max |r_rms - r0 |cos(w_b t)|| / r0 = %.2e' % (abs(rr-rr[0]*abs(np.cos(w*t))).max()/rr[0]))"
}

sec4() {
    echo "== 4. witness energy gain in a rigid-driver blowout =="
    $QS gain.in time.steps=$(q 20 8) driver.nparticles=$(q 500000 100000) > /dev/null
    python3 -c "
import sys; sys.path.insert(0,'../tools')
from quarz_read import read_beamlog
import numpy as np
b=read_beamlog('out_gain/beams.txt')['witness']
print('d<gamma>/dt = %.4f   (-Ez at xi=7 from the blowout run: 0.4471)' % np.polyfit(b['t'],b['gamma_mean'],1)[0])"
}

sec5() {
    echo "== 5. pinched witness + ion motion: grid comparison (reference takes ~1 min) =="
    im() { name=$1; shift; $QS ../examples/pwfa_ion_motion.in time.steps=0 output.dir=out_im_$name "$@" > /dev/null; }
    if [ "$QUICK" = 0 ]; then
        im stretched
        im u001 grid.type=uniform grid.dr=0.01
        im u0005 grid.type=uniform grid.dr=0.005
        im u0001 grid.type=uniform grid.dr=0.00025 xi.max=7.3
        python3 im_compare.py
    else
        im stretched xi.max=7.3
        im u001 grid.type=uniform grid.dr=0.01 xi.max=7.3
        im qref xi.max=7.3 "grid.regions=0.05:0.000125 2.5:0.0025 8.0:0.0125"
        python3 im_compare.py qref u001 stretched
    fi
}

sec6() {
    echo "== 6. azimuthal mode 1: displaced driver vs translated axisymmetric solution =="
    $QS linear.in xi.max=12 driver.analytic=1 driver.n0=0.01 output.dir=out_m1_ref > /dev/null
    $QS linear.in xi.max=12 driver.analytic=1 driver.n0=0.01 modes=1 driver.x0=0.01 output.dir=out_m1_shift > /dev/null
    echo " linear regime (d = 0.01):"
    python3 check_shift2.py out_m1_ref out_m1_shift 0.01 3 -1e9 1e9 0.15 2>/dev/null
    [ "$QUICK" = 1 ] && { rm -rf out_m1_*; return 0; }
    B="blowout.in xi.max=9 xi.step=0.005 grid.type=regions grid.max_ratio=1.05 driver.analytic=1"
    $QS $B "grid.regions=0.5:0.0025 2.5:0.01 8:0.05" output.dir=out_m1_bref > /dev/null
    $QS $B "grid.regions=0.5:0.0025 2.5:0.01 8:0.05" modes=1 driver.x0=0.002 output.dir=out_m1_bshift > /dev/null
    echo " blowout, inside the first bucket (d = 0.002):"
    python3 check_shift2.py out_m1_bref out_m1_bshift 0.002 1.0 0 8 0.15 2>/dev/null
    rm -rf out_m1_ref out_m1_shift out_m1_bref out_m1_bshift   # large files
}

sec7() {
    echo "== 7. azimuthal mode 1: centroid betatron oscillation of an offset beam =="
    $QS betatron.in modes=1 witness.x0=0.05 witness.nsym=4 output.every=0 output.dir=out_m1_cbt > /dev/null
    python3 -c "
import sys; sys.path.insert(0,'../tools')
from quarz_read import read_beamlog
import numpy as np
b=read_beamlog('out_m1_cbt/beams.txt')['witness']; t=b['t']; w=1/np.sqrt(2*1000); xc=b['x_mean']
print('max |x_c - x0 cos(w_b t)|/x0 = %.2e,  max |y_c|/x0 = %.1e' % (abs(xc-xc[0]*np.cos(w*t)).max()/xc[0], abs(b['y_mean']).max()/xc[0]))"
}

sec8() {
    echo "== 8. function parser: parsed profiles vs built-in ones (must agree to round-off) =="
    B="blowout.in xi.max=9 driver.analytic=1"
    $QS $B output.dir=out_p_ref > /dev/null
    $QS $B "plasma.density(x,y,z)=1" output.dir=out_p_1 > /dev/null
    python3 cmp_fields.py out_p_ref out_p_1 'uniform: built-in vs parsed "1"' 8.5
    $QS $B plasma.radial=channel plasma.channel_depth=0.3 plasma.channel_radius=1.5 output.dir=out_p_c1 > /dev/null
    $QS $B my_constants.dn=0.3 my_constants.rc=1.5 "plasma.density(x,y,z)=1 + dn*(x^2+y^2)/rc^2" output.dir=out_p_c2 > /dev/null
    python3 cmp_fields.py out_p_c1 out_p_c2 "channel: built-in vs parsed" 8.5
    $QS $B driver.cut=99 driver.profile=parsed "driver.density(x,y,xi)=10*exp(-(x^2+y^2)/(2*0.3^2)-(xi-3)^2/2)" output.dir=out_p_b > /dev/null
    python3 cmp_fields.py out_p_ref out_p_b "analytic driver: built-in vs parsed" 8.5
    rm -rf out_p_*
}

sec9() {
    echo "== 9. MPI (decomposition along xi) vs serial with beams.xi_shape = ngp =="
    if command -v mpirun > /dev/null 2>&1; then
        MPIRUN="mpirun --oversubscribe -np"
        [ "$(id -u)" = "0" ] && MPIRUN="mpirun --allow-run-as-root --oversubscribe -np"
        export OMP_NUM_THREADS=1
        G="gain.in driver.nparticles=$(q 500000 100000) time.steps=$(q 20 4) output.every=$(q 10 2) output.beam_every=$(q 10 2) output.beam_slices=50"
        $QS $G beams.xi_shape=ngp output.dir=out_mpi_s > /dev/null
        $MPIRUN 2 $QS $G output.dir=out_mpi_p2 > /dev/null
        echo " witness gain, 2 ranks (expected: bit-identical, 0):"
        python3 cmp_runs.py out_mpi_s out_mpi_p2 0 || true
        $QS mpi_slip.in time.steps=$(q 24 8) beams.xi_shape=ngp output.dir=out_mpi_s2 > /dev/null
        $MPIRUN 3 $QS mpi_slip.in time.steps=$(q 24 8) output.dir=out_mpi_p3 > /dev/null
        echo " slipping beam + mobile ions + m = 1, 3 ranks (expected: round-off from particle order):"
        python3 cmp_runs.py out_mpi_s2 out_mpi_p3 1e-6 || true
        rm -rf out_mpi_*
    else
        echo " (mpirun not found - skipped)"
    fi
}

sec10() {
    echo "== 10. openPMD output (needs openPMD-api in the build and the openpmd-api Python module) =="
    if python3 -c "import openpmd_api" 2> /dev/null; then
    export OMP_NUM_THREADS=1   # deterministic deposits: bit-identical comparisons
    H="../examples/hosing.in time.steps=$(q 4 2) output.every=2 output.beam_every=2 driver.nparticles=$(q 40000 8000) xi.max=$(q 6 4) beams.xi_shape=ngp output.format=both units.n0_cm3=1e17"
        if $QS $H output.openpmd_grid=native output.dir=out_opmd_nat > out_opmd.log 2>&1; then
            echo " native-node openPMD vs native files (expected 0):"; python3 check_openpmd.py out_opmd_nat native | tail -1
            if python3 -c "import vtk" 2> /dev/null; then
                python3 ../tools/paraview/make_rectilinear_testfiles.py out_opmd_pv > /dev/null
                F=$(ls out_opmd_nat/openpmd/*.h5 | head -1)
                echo " ParaView openPMD reader with rectilinear meshes (tools/paraview, vtk module; expected PASS):"
                python3 ../tools/paraview/test_openpmd_reader.py ../tools/paraview/openPMDRectilinearReader.py \
                    "quarz:$F" cart:out_opmd_pv/cart3d_xyz.h5 cart:out_opmd_pv/cart3d_zyx.h5 cart:out_opmd_pv/cart2d_xz.h5 \
                    theta:out_opmd_pv/theta_dataset.h5 regular:out_opmd_pv/cart3d_regular.h5 2> /dev/null | grep -E "PASS|FAIL"
            fi
            $QS $H output.dir=out_opmd_uni > /dev/null 2>&1
            echo " uniform openPMD grid vs interpolated native data (expected round-off):"; python3 check_openpmd.py out_opmd_uni uniform | tail -1
            if command -v mpirun > /dev/null 2>&1; then
                MPIRUN="mpirun --oversubscribe -np"; [ "$(id -u)" = "0" ] && MPIRUN="mpirun --allow-run-as-root --oversubscribe -np"
                OMP_NUM_THREADS=1 $MPIRUN 2 $QS $H output.format=openpmd output.dir=out_opmd_p2 > /dev/null 2>&1
                echo " 2 MPI ranks vs serial (expected 0):"; python3 cmp_openpmd.py out_opmd_uni out_opmd_p2
                OMP_NUM_THREADS=1 $MPIRUN 2 $QS $H output.format=openpmd output.openpmd_grid=native output.dir=out_opmd_p2n > /dev/null 2>&1
                echo " 2 MPI ranks vs serial, native nodes (expected 0):"; python3 cmp_openpmd.py out_opmd_nat out_opmd_p2n
            fi
        else
            echo " (this build has no openPMD support - skipped)"
        fi
        rm -rf out_opmd_* out_opmd.log
    else
        echo " (python module openpmd_api not found - skipped)"
    fi
}

sec11() {
    echo "== 11. laser envelope solver =="
    $QS laser_vacuum.in > /dev/null
    echo " diffraction in vacuum (k0 = 20, w0 = 2, through the focus):"; python3 laser_check.py vacuum out_laser_vac 20 2 40 0.1
    $QS laser_linear.in > /dev/null
    echo " linear wake (a0 = 0.05) vs (d^2/dxi^2 + 1) psi = <a^2>/2 (difference = O(a0^2) nonlinearity):"; python3 laser_check.py linear out_laser_lin 0.05 3 1.4142135623730951 4
    $QS laser_channel.in time.dt=$(q 2.5 5) time.steps=$(q 120 60) > /dev/null
    echo " matched parabolic channel (stretched grid):"; python3 laser_check.py channel out_laser_ch 10 3
    [ "$QUICK" = 1 ] && { rm -rf out_laser_*; return 0; }
    S="laser_selffocus.in grid.rmax=90 grid.regions=40:0.2 90:0.5 xi.step=0.1 time.dt=100 time.steps=40 laser.w0=20"
    $QS $S laser.a0=0.4 electrons.density=1e-9 output.dir=out_laser_sf0 > /dev/null
    $QS $S laser.a0=0.2 output.dir=out_laser_sf05 > /dev/null
    $QS $S laser.a0=0.4 output.dir=out_laser_sf2 > /dev/null
    echo " relativistic self-focusing, P/Pc = 0 (vacuum: 1 .97 .894 .8 .707), 0.5, 2:"
    python3 laser_check.py selffocus out_laser_sf0 0.4 out_laser_sf05 0.2 out_laser_sf2 0.4
    rm -rf out_laser_*
}

sec12() {
    echo "== 12. ionization =="
    $QS ion_adk.in > /dev/null
    echo " field (ADK) ionization of H and He by a beam field: code vs independent ADK (Markov chain of the per-slice probabilities):"
    python3 ion_check.py adk out_ion_adk 1e17 16 0.01 gas:He:1e-4 hyd:H:1e-4
    $QS ion_laser.in gas.ppc=$(q 96 16) > /dev/null
    echo " laser ionization of N (up to N5+), a0 = 0.1, 0.8 um at 1e17 cm^-3, wide spot: code vs cycle-resolved ADK and test particles:"
    python3 ion_check.py laser out_ion_laser 1e17 132 0.1 3 10 N 1e-6 5 2>/dev/null
    $QS ion_impact.in gas.ppc=$(q 32 8) gas.ionization.nsplit=$(q 2000 200) > /dev/null
    echo " impact ionization of Ar by a rigid 400 GeV proton beam (Bethe, M^2 = 4.22, C = 37.93; nsplit = 2000):"
    python3 ion_check.py impact out_ion_impact 1e15 4.22:37.93 10 1 15 50 427 1
    $QS ion_impact.in gas.ppc=$(q 32 8) gas.ionization.nsplit=$(q 2000 200) gas.ionization.quiet=1 output.dir=out_ion_impact_q > /dev/null
    echo " the same with gas.ionization.quiet = 1 (quanta emitted by an accumulator, interleaved phases):"
    python3 ion_check.py impact out_ion_impact_q 1e15 4.22:37.93 10 1 15 50 427 1
    $QS ion_adk.in gas.ionization.quiet=1 hyd.ionization.quiet=1 output.dir=out_ion_adk_q > /dev/null
    echo " ADK of H and He with ionization.quiet = 1 (whole particles: exponential thresholds of the optical depth):"
    python3 ion_check.py adk out_ion_adk_q 1e17 16 0.01 gas:He:1e-4 hyd:H:1e-4
    local G=""; for s in he ne ar kr xe; do G="$G $s.ppc=$(q 32 16) $s.ionization.nsplit=$(q 2000 1000)"; done
    $QS ion_impact_gases.in $G xi.step=$(q 0.1 0.25) > /dev/null
    echo " built-in Bethe data (Rieke & Prepejchal): impact ionization of dilute He, Ne, Ar, Kr, Xe by a rigid beam at"
    echo " gamma = 3.39 (minimum) vs the measured minimum cross-sections (Blum, Riegler & Rolandi, Table 1.2, +-4 %):"
    python3 ion_check.py impact_gases out_ion_gases 1e16 40 1 15 50 3.39 he:1e-6:18.6:5.0 ne:1e-6:43.3:12.4 \
        ar:1e-6:90.3:25-28 kr:1e-6:-:- xe:1e-6:172:41-44
    rm -rf out_ion_*
}

sec13() {
    echo "== 13. adaptive sub-slicing of the plasma push (pusher.max_cells_per_step) =="
    # paper case (pinched witness, mobile ions, axis cell 5e-4), box to xi = 12, dxi = 0.01
    local P="../paper/inputs/conv_s1.in xi.max=12 output.field_files=0"
    $QS $P xi.step=$(q 0.000625 0.0025) output.dir=out_ss_ref > out_ss_ref.log
    $QS $P xi.step=0.01 output.dir=out_ss_plain > out_ss_plain.log
    $QS $P xi.step=0.01 pusher.max_cells_per_step=1 output.dir=out_ss_sub > out_ss_sub.log
    echo " dxi = 0.01 without / with sub-slicing vs reference dxi = $(q 0.000625 0.0025) (relative r.m.s. errors on the axis):"
    python3 subslice_check.py out_ss_ref out_ss_plain out_ss_sub
    if [ -n "$MPIRUN" ]; then
        local S="$P xi.step=0.01 pusher.max_cells_per_step=1 beams.xi_shape=ngp xi.max=$(q 12 10)"
        OMP_NUM_THREADS=1 $QS $S output.dir=out_ss_s > /dev/null
        OMP_NUM_THREADS=1 $MPIRUN 2 $QS $S output.dir=out_ss_p2 > /dev/null
        echo " with sub-slicing, 2 MPI ranks vs serial (expected: bit-identical, 0):"
        python3 cmp_runs.py out_ss_s out_ss_p2 0 || true
    fi
    rm -rf out_ss_*
}

sec14() {
    echo "== 14. regularization of the bubble back: radial smoothing of the plasma sources =="
    # paper case (pinched witness, mobile ions, axis cell 5e-4), box to xi = 12. Criterion: the fields
    # AFTER the closure spike converge in dxi and in the axis cell (the spike height need not).
    local P="../paper/inputs/conv_s1.in xi.max=12 output.field_files=0" S="plasma.smooth_length=0.005"
    local XR=$(q 0.000625 0.00125) G="2.5:0.01 8.0:0.05" d
    $QS $P xi.step=$XR output.dir=out_sm_cold_ref > /dev/null
    $QS $P $S xi.step=$XR output.dir=out_sm_a5e-3_ref > /dev/null
    for d in 0.005 0.0025 $(q 0.00125 ""); do
        $QS $P xi.step=$d output.dir=out_sm_cold_dxi$d > /dev/null
        $QS $P $S xi.step=$d output.dir=out_sm_a5e-3_dxi$d > /dev/null
    done
    for h in 0.001 0.00025; do
        $QS $P $S xi.step=$(q 0.0025 0.005) "grid.regions=0.05:$h $G" output.dir=out_sm_a5e-3_h0=$h > /dev/null
    done
    echo " cold plasma, xi step vs reference dxi = $XR:"
    python3 smoothing_check.py out_sm_cold_ref out_sm_cold_dxi*
    echo " a = 0.005, xi step and axis cell h0 vs reference dxi = $XR, h0 = 5e-4:"
    python3 smoothing_check.py out_sm_a5e-3_ref out_sm_a5e-3_dxi* out_sm_a5e-3_h0=*
    echo " model difference a = 0.005 vs cold (both references; O(a)):"
    python3 smoothing_check.py out_sm_cold_ref out_sm_a5e-3_ref
    if [ -n "$MPIRUN" ]; then
        local M="$P $S xi.step=0.01 beams.xi_shape=ngp xi.max=$(q 12 10)"
        OMP_NUM_THREADS=1 $QS $M output.dir=out_sm_s > /dev/null
        OMP_NUM_THREADS=1 $MPIRUN 2 $QS $M output.dir=out_sm_p2 > /dev/null
        echo " with smoothing, 2 MPI ranks vs serial (expected: bit-identical, 0):"
        python3 cmp_runs.py out_sm_s out_sm_p2 0 || true
    fi
    rm -rf out_sm_*
}

sec15() {
    echo "== 15. checkpoints / restart and diagnostic groups =="
    # restart: uninterrupted run A (checkpoint every 2 steps) vs run B restarted from step 2;
    # one OpenMP thread (with more, the atomic deposits are not bit-reproducible anyway)
    local E="../paper/inputs/evo_s1.in time.steps=$(q 6 4) output.every=1 output.beam_every=1 beams.xi_shape=ngp xi.step=$(q 0.005 0.01)"
    local C="checkpoint.every=2 checkpoint.keep=0" R="restart.from=out_rs_A/checkpoints/chk_000002"
    local L="../examples/lwfa_laser.in time.steps=$(q 4 2) output.every=1 output.beam_every=1 beams.xi_shape=ngp xe.ionization.born_dump=0"
    local O="../paper/inputs/evo_s1.in time.steps=2 output.every=2 output.beam_every=2 beams.xi_shape=ngp xi.step=$(q 0.005 0.01)"
    export OMP_NUM_THREADS=1
    $QS $E $C output.dir=out_rs_A > /dev/null
    $QS $E $R output.dir=out_rs_B > /dev/null
    echo " particle witness, mobile ions, serial restart (expected: bit-identical):"
    python3 restart_check.py out_rs_A out_rs_B 0 || true
    $QS $L checkpoint.every=1 checkpoint.keep=0 output.dir=out_rs_LA > /dev/null
    $QS $L restart.from=out_rs_LA/checkpoints/chk_000001 output.dir=out_rs_LB > /dev/null
    echo " laser envelope + witness, serial restart (expected: bit-identical):"
    python3 restart_check.py out_rs_LA out_rs_LB 0 || true
    if [ -n "$MPIRUN" ]; then
        $MPIRUN 2 $QS $E $C output.dir=out_rs_P2A > /dev/null
        $MPIRUN 2 $QS $E restart.from=out_rs_P2A/checkpoints/chk_000002 output.dir=out_rs_P2B > /dev/null
        echo " 2 ranks, restart with 2 ranks (expected: bit-identical):"
        python3 restart_check.py out_rs_P2A out_rs_P2B 0 || true
        $MPIRUN 3 $QS $E restart.from=out_rs_P2A/checkpoints/chk_000002 output.dir=out_rs_P3B > /dev/null
        echo " 2 ranks, restart with 3 ranks (particles redistributed; expected: <= round-off):"
        python3 restart_check.py out_rs_P2A out_rs_P3B 1e-10 || true
        $MPIRUN 2 $QS $L checkpoint.every=1 checkpoint.keep=0 output.dir=out_rs_L2A > /dev/null
        $MPIRUN 3 $QS $L restart.from=out_rs_L2A/checkpoints/chk_000001 output.dir=out_rs_L3B > /dev/null
        echo " laser, 2 ranks, restart with 3 ranks (envelope slices redistributed; expected: <= round-off):"
        python3 restart_check.py out_rs_L2A out_rs_L3B 1e-10 || true
    fi
    # restart into the same directory: the logs are truncated at the restart step and continued
    cp -r out_rs_A out_rs_C
    $QS $E restart.from=out_rs_C/checkpoints/chk_000002 output.dir=out_rs_C > /dev/null
    if diff -q out_rs_A/beams.txt out_rs_C/beams.txt > /dev/null; then
        echo " same-directory restart: beams.txt identical to the uninterrupted run  OK"
    else echo " same-directory restart: beams.txt differs  FAIL"; fi
    # pruning: keep = 2 leaves the two newest complete checkpoints
    $QS $E checkpoint.every=1 checkpoint.keep=2 output.every=0 output.beam_every=0 output.dir=out_rs_K > /dev/null
    echo " checkpoint.keep = 2, every step: kept $(ls out_rs_K/checkpoints | tr '\n' ' ')(expected: the two newest)"
    # diagnostic groups: exact subsets of the main output
    $QS $O diag.names=light diag.light.every=2 "diag.light.fields=ez psi ne" diag.light.rmax=1.0 diag.light.xi_stride=3 \
        diag.light.beams=witness diag.light.particle_stride=7 output.dir=out_grp > /dev/null
    echo " diagnostic group (ez psi ne, r <= 1, every 3rd slice, witness every 7th particle), serial:"
    python3 group_check.py out_grp out_grp/light ez,psi,ne 1.0 3 7 witness || true
    if [ -n "$MPIRUN" ]; then
        $MPIRUN 3 $QS $O diag.names=light diag.light.every=2 "diag.light.fields=ez bth" diag.light.rmax=0.5 \
            diag.light.xi_stride=4 diag.light.beams=none output.dir=out_grp3 > /dev/null
        echo " diagnostic group (ez bth, r <= 0.5, every 4th slice), 3 ranks:"
        python3 group_check.py out_grp3 out_grp3/light ez,bth 0.5 4 1 none || true
    fi
    if $QS ../paper/inputs/evo_s1.in time.steps=0 output.every=1 diag.names=pmd diag.pmd.every=1 diag.pmd.format=openpmd \
         diag.pmd.openpmd_backend=json "diag.pmd.fields=ez psi" diag.pmd.beams=witness diag.pmd.particle_stride=5 \
         diag.pmd.xi_stride=2 diag.pmd.rmax=0.5 output.dir=out_grpo > /dev/null 2>&1; then
        echo " openPMD group (E_z, psi, witness every 5th particle, every 2nd slice):"
        python3 group_openpmd_check.py out_grpo/pmd/openpmd/data_000000.json 0 E.z,psi witness 901 40000 || true
    fi
    unset OMP_NUM_THREADS
    rm -rf out_rs_* out_grp*
}

sec16() {
    echo "== 16. adaptive time step =="
    local T=$(q 3000 1200)
    $QS adaptive_ramp.in time.t_end=$T > /dev/null
    echo " ion channel, density 1 -> 4, gamma 1000, 128 steps per betatron period:"
    python3 adaptive_check.py ramp out_adapt_ramp 128 1000 1.5 0 3 $T
    local D="adaptive_decel.in time.t_end=$(q 100 60)"
    $QS $D time.adaptive_lag=1 output.dir=out_ad_l1 > /dev/null
    $QS $D time.adaptive_lag=6 output.dir=out_ad_l6 > /dev/null
    echo " decelerating witness (gamma 50 -> 6), 32 steps per period, gamma known 1 and 6 steps late:"
    python3 adaptive_check.py safe out_ad_l1 32 1
    python3 adaptive_check.py safe out_ad_l6 32 6
    python3 adaptive_check.py compare out_ad_l6 out_ad_l1
    # MPI and restart: one OpenMP thread (bit-identity)
    export OMP_NUM_THREADS=1
    local M="adaptive_decel.in time.t_end=40 output.every=5 output.beam_every=5"
    $QS $M time.adaptive_lag=3 checkpoint.every=6 checkpoint.keep=0 checkpoint.dir=out_ad_chkA output.dir=out_ad_A > /dev/null
    $QS $M time.adaptive_lag=3 restart.from=out_ad_chkA/chk_000006 output.dir=out_ad_B > /dev/null
    echo " serial restart (expected: bit-identical):"
    python3 restart_check.py out_ad_A out_ad_B 0 || true
    if [ -n "$MPIRUN" ]; then
        $MPIRUN 2 $QS $M checkpoint.every=6 checkpoint.keep=0 checkpoint.dir=out_ad_chkP output.dir=out_ad_PA > /dev/null
        echo " 2 ranks (lag 3) vs serial with lag 3 (expected: fields and particles bit-identical, beam log sums round-off; same time steps):"
        python3 cmp_runs.py out_ad_A out_ad_PA 1e-12 || true
        if diff <(grep -v "^#" out_ad_A/timestep.txt) <(grep -v "^#" out_ad_PA/timestep.txt) > /dev/null; then
            echo "  timestep.txt identical  OK"; else echo "  timestep.txt differs  FAIL"; fi
        $MPIRUN 2 $QS $M restart.from=out_ad_chkP/chk_000006 output.dir=out_ad_PB > /dev/null
        echo " 2 ranks, restart (expected: bit-identical):"
        python3 restart_check.py out_ad_PA out_ad_PB 0 || true
    fi
    unset OMP_NUM_THREADS
    rm -rf out_adapt_ramp out_ad_*
}

sec17() {
    echo "== 17. GUI server (tools/quarz_gui.py, no browser) =="
    # QUARZ_TEST_REMOTE=1: also Slurm / ssh-to-localhost runs (not in the references)
    OMP_NUM_THREADS=1 python3 gui_check.py betatron.in $QS $(( 20000 + $$ % 20000 )) ${QUARZ_TEST_REMOTE:+--remote}
    rm -rf out_gui out_gui_* quarz_gui_*.out quarz_gui_job.sh quarz_gui_ssh.*
}

sec18() {
    echo "== 18. multiple Coulomb scattering of beam particles on the plasma =="
    # one thread: the random numbers hash the particle state, which round-off of threaded deposits changes
    export OMP_NUM_THREADS=1
    local NP=$(q 100000 40000)
    $QS scatter_free.in witness.nparticles=$NP > /dev/null
    python3 scatter_check.py free out_scat_free 20 100 10 10 1 1 1e6
    $QS scatter_free.in witness.nparticles=$NP scattering.Z=3 scattering.ion_charge=1 scattering.coulomb_log_electrons=7 \
        output.dir=out_scat_free3 > /dev/null
    python3 scatter_check.py free out_scat_free3 20 100 10 7 3 1 1e6
    $QS scatter_channel.in time.steps=$(q 200 80) witness.nparticles=$NP > /dev/null
    python3 scatter_check.py channel out_scat_ch 1000 10 1e5
    local M="scatter_channel.in time.steps=10 output.beam_every=5 output.every=5 beams.xi_shape=ngp witness.nparticles=20000"
    $QS $M checkpoint.every=4 checkpoint.keep=0 checkpoint.dir=out_sc_chk output.dir=out_sc_A > /dev/null
    $QS $M restart.from=out_sc_chk/chk_000004 output.dir=out_sc_B > /dev/null
    echo " restart (expected: bit-identical, the random numbers depend on the particle state, not its index):"
    python3 restart_check.py out_sc_A out_sc_B 0 || true
    if [ -n "$MPIRUN" ]; then
        $MPIRUN 2 $QS $M output.dir=out_sc_P > /dev/null
        echo " 2 ranks vs serial (expected: particles bit-identical; fields at round-off, as without scattering):"
        python3 cmp_runs.py out_sc_A out_sc_P 1e-14 || true
    fi
    unset OMP_NUM_THREADS
    rm -rf out_scat_* out_sc_*
}

sec19() {
    echo "== 19. two-stage runs: driver box -> rear boundary file -> witness box =="
    export OMP_NUM_THREADS=1
    local S1="xi.max=5 beams=driver" S2="xi.min=5.01 beams=witness"
    local N="driver.nparticles=$(q 200000 50000)"
    # (a) same time step: the stage-2 box reproduces the single run exactly
    $QS boundary.in $N output.dir=out_bnd_A > /dev/null
    $QS boundary.in $N $S1 boundary.write=out_bnd_a.bin output.dir=out_bnd_A1 > /dev/null
    $QS boundary.in $N $S2 boundary.read=out_bnd_a.bin output.dir=out_bnd_A2 > /dev/null
    echo " (a) rigid driver, same dt (expected: bit-identical):"
    python3 bnd_check.py fields out_bnd_A out_bnd_A2 6 6
    # (b) soft witness (gamma 30) with dt/4 in stage 2; rigid driver -> the boundary state is constant
    local W="witness.gamma=30 witness.emit_n=0.01 witness.sigma_r=0.1"
    $QS boundary.in $N $W time.dt=2.5 time.steps=24 output.every=24 output.beam_every=24 output.dir=out_bnd_B > /dev/null
    $QS boundary.in $N $W $S2 boundary.read=out_bnd_a.bin time.dt=2.5 time.steps=24 output.every=24 output.beam_every=24 \
        output.dir=out_bnd_B2 > /dev/null
    echo " (b) stage 2 with dt/4 between the stage-1 steps, rigid driver (expected: bit-identical):"
    python3 bnd_check.py fields out_bnd_B out_bnd_B2 24 24
    # (c) evolving driver (gamma 300): stage 1 with dt = 10, stage 2 with dt = 2.5, vs single runs
    local C="$W driver.rigid=0 driver.gamma=300 driver.emit_n=0.3"
    $QS boundary.in $N $C time.dt=2.5 time.steps=24 output.every=24 output.beam_every=24 output.dir=out_bnd_ref > /dev/null
    $QS boundary.in $N $C time.dt=10 time.steps=6 output.every=6 output.beam_every=6 output.dir=out_bnd_coarse > /dev/null
    $QS boundary.in $N $C $S1 boundary.write=out_bnd_c.bin output.dir=out_bnd_C1 > /dev/null
    $QS boundary.in $N $C $S2 boundary.read=out_bnd_c.bin time.dt=2.5 time.steps=24 output.every=24 output.beam_every=24 \
        output.dir=out_bnd_two > /dev/null
    echo " (c) evolving driver: single run dt = 2.5 (reference) vs two-stage (driver dt = 10, witness dt = 2.5) vs single dt = 10:"
    python3 bnd_check.py witness out_bnd_ref out_bnd_two out_bnd_coarse
    if [ -n "$MPIRUN" ]; then
        $MPIRUN 2 $QS boundary.in $N $S1 boundary.write=out_bnd_p.bin output.dir=out_bnd_P1 > /dev/null
        $MPIRUN 2 $QS boundary.in $N $S2 boundary.read=out_bnd_p.bin output.dir=out_bnd_P2 > /dev/null
        echo " (a) with 2 ranks in each stage (expected: bit-identical to the single serial run):"
        python3 bnd_check.py fields out_bnd_A out_bnd_P2 6 6
    fi
    unset OMP_NUM_THREADS
    rm -rf out_bnd_*
}

sec20() {
    echo "== 20. beam spin (Thomas-BMT) =="
    export OMP_NUM_THREADS=1
    # (a) pure ion channel: s_r = sin(kappa theta), electrons, muons, antiprotons
    $QS spin.in time.steps=$(q 200 100) output.beam_every=$(q 200 100) > /dev/null
    echo " (a) ion channel, analytic s_r = sin(kappa theta), kappa = gamma beta^2 (a + 1/(gamma+1)):"
    python3 spin_check.py channel out_spin $(q 200 100)
    # (b) gamma 3 witness in a wake (E_r, E_z, B_theta), a = 2 to make the B term count; reference RK4
    local NS=$(q 80 40) DT=$(q 0.5 1)
    $QS spin_wake.in witness.anomalous_moment=2 time.dt=$DT time.steps=$NS output.every=$NS output.beam_every=$NS > /dev/null
    echo " (b) wake, vs an RK4 integration of Lorentz + T-BMT in the stored fields (error ~ dt^2):"
    python3 spin_check.py wake out_spin_wake 0 $NS $DT 2
    # (c) multiple Coulomb scattering: s_perp = kappa theta_perp to first order
    $QS scatter_free.in "witness.spin=0 0 1" witness.nparticles=$(q 40000 10000) output.dir=out_spin_sc > /dev/null
    echo " (c) scattering only (expected slopes 1 - O((kappa theta)^2)):"
    python3 spin_check.py free out_spin_sc 20 0.00115965218128
    # (d) restart and 2 ranks bit-identical
    local M="spin.in time.steps=10 output.beam_every=5 output.every=5 beams.xi_shape=ngp xe.ionization.born_dump=0"
    $QS $M checkpoint.every=4 checkpoint.keep=0 checkpoint.dir=out_spin_chk output.dir=out_spin_A > /dev/null
    $QS $M restart.from=out_spin_chk/chk_000004 output.dir=out_spin_B > /dev/null
    echo " (d) restart (expected: bit-identical):"
    python3 restart_check.py out_spin_A out_spin_B 0 || true
    if [ -n "$MPIRUN" ]; then
        $MPIRUN 2 $QS $M output.dir=out_spin_P > /dev/null
        echo " 2 ranks vs serial (expected: particles and spins bit-identical; fields, beams.txt moments at round-off):"
        python3 cmp_runs.py out_spin_A out_spin_P 1e-8 || true
    fi
    unset OMP_NUM_THREADS
    rm -rf out_spin*
}

sec21() {
    echo "== 21. trapped plasma electrons -> beam particles (Trojan-horse ionization injection) =="
    local N=$(q 30 10) N2=$(q 60 20)
    $QS trap.in time.steps=$N output.every=$N output.beam_every=$N > /dev/null
    echo " (a) H = gamma - p_z - psi = 1 - psi(birth), conserved through the conversion and the beam push:"
    python3 trap_check.py h out_trap $N
    $QS trap.in time.dt=0.1 time.steps=$N2 output.every=$N2 output.beam_every=$N2 output.dir=out_trap_dt > /dev/null
    echo " (b) trapped charge per unit time, dt = 0.2 vs 0.1 (expected: equal within the injection noise):"
    python3 trap_check.py rate out_trap $N 0.2 out_trap_dt $N2 0.1
    $QS trap.in time.steps=$N output.every=$N output.beam_every=$N pusher.max_qsa_factor=20 output.dir=out_trap_20 > /dev/null
    $QS trap.in time.steps=$N output.every=$N output.beam_every=$N pusher.max_qsa_factor=70 output.dir=out_trap_70 > /dev/null
    echo " (c) conversion threshold gamma/Delta = 35, 20, 70 (expected: the same trapped charge):"
    python3 trap_check.py thr out_trap out_trap_20 out_trap_70 $N
    # (d) with beam loading (not passive, 1 % helium): restart and 2 ranks bit-identical
    export OMP_NUM_THREADS=1
    local M="trap.in time.steps=10 output.beam_every=5 output.every=5 beams.xi_shape=ngp hel_trapped.passive=0 he.density=0.01"
    $QS $M checkpoint.every=4 checkpoint.keep=0 checkpoint.dir=out_trap_chk output.dir=out_trap_A > /dev/null
    $QS $M restart.from=out_trap_chk/chk_000004 output.dir=out_trap_B > /dev/null
    echo " (d) with beam loading: restart (expected: bit-identical):"
    python3 restart_check.py out_trap_A out_trap_B 0 || true
    if [ -n "$MPIRUN" ]; then
        $MPIRUN 2 $QS $M output.dir=out_trap_P > /dev/null
        echo " 2 ranks vs serial (expected: bit-identical):"
        python3 cmp_runs.py out_trap_A out_trap_P 1e-14 || true
    fi
    unset OMP_NUM_THREADS
    rm -rf out_trap*
}

sec22() {
    echo "== 22. delta electrons of beam impact ionization (<species>.impact.secondary_energy) =="
    local D="ion_delta.in xe.ionization.nsplit=$(q 100000 20000) xi.step=$(q 0.1 0.25)"
    $QS $D > /dev/null
    local XE="8.04 72.35 34561:2,5453:2,5107:2,4786:4,1148.7:2,1002.1:2,940.6:4,689.0:4,676.4:6"
    echo " (a) dilute Xe, protons at gamma = 3.39: energy spectrum (J = 8.7 eV + free-electron tail of the n = 1-3 subshells)"
    echo "     and binary-collision angle vs the model:"
    python3 delta_check.py spectrum out_delta xe 8.7 1836.15 1 3.39 kinematic $XE
    $QS $D xe.impact.inner_shells_eV="100 1000" xe.impact.inner_shells_n="50 50" output.dir=out_delta_sh > /dev/null
    echo " (a2) artificial subshells (100 eV, 1 keV; 50 electrons each) to resolve the mixture:"
    python3 delta_check.py spectrum out_delta_sh xe 8.7 1836.15 1 3.39 kinematic 8.04 72.35 100:50,1000:50
    $QS $D b.mass=1 b.charge=-1 b.gamma=20 xe.impact.J_eV=2e6 output.dir=out_delta_e > /dev/null
    echo " (b) electrons at gamma = 20 (W_max = (gamma - 1)/2, Moller kinematics), J = 2 MeV to resolve the cut-off:"
    python3 delta_check.py spectrum out_delta_e xe 2e6 1 -1 20 kinematic $XE
    $QS $D xe.impact.secondary_angle=isotropic output.dir=out_delta_iso > /dev/null
    echo " (c) isotropic emission:"
    python3 delta_check.py spectrum out_delta_iso xe 8.7 1836.15 1 3.39 isotropic $XE
    $QS $D b.gamma=427 xe.impact.J_eV=5.11e6 output.dir=out_delta_fast > /dev/null
    echo " (d) 400 GeV protons, J = 5.11 MeV: fast delta electrons (gamma/Delta > 35) -> beam particles of electrons_trapped:"
    python3 delta_check.py trapped out_delta_fast xe 35
    export OMP_NUM_THREADS=1
    $QS $D xi.step=0.25 output.dir=out_delta_on > /dev/null
    $QS $D xi.step=0.25 xe.impact.secondary_energy=0 output.dir=out_delta_off > /dev/null
    echo " (e) the ionization itself does not change:"
    python3 delta_check.py same out_delta_on out_delta_off xe
    # (f) fast deltas as non-passive beam electrons that ionize the gas themselves: restart and MPI bit-identical
    local M="$D b.gamma=427 xe.impact.J_eV=5.11e6 xi.step=0.25 time.steps=$(q 6 4) output.every=2 output.beam_every=2 beams.xi_shape=ngp xe.ionization.born_dump=0"
    $QS $M checkpoint.every=2 checkpoint.keep=0 checkpoint.dir=out_delta_chk output.dir=out_delta_A > /dev/null
    $QS $M restart.from=out_delta_chk/chk_000002 output.dir=out_delta_B > /dev/null
    echo " (f) with fast deltas ionizing as beam electrons: restart (expected: bit-identical):"
    python3 restart_check.py out_delta_A out_delta_B 0 || true
    if [ -n "$MPIRUN" ]; then
        $MPIRUN 2 $QS $M output.dir=out_delta_P > /dev/null
        echo " 2 ranks vs serial (expected: bit-identical):"
        python3 cmp_runs.py out_delta_A out_delta_P 1e-14 || true
    fi
    unset OMP_NUM_THREADS
    rm -rf out_delta*
}

sec23() {
    echo "== 23. electron-impact ionization by plasma electrons (<species>.ionization.electron_impact) =="
    local E="eimpact.in xe.ppc=$(q 64 16) xe.ionization.born_dump=0"
    echo " (a) dilute gas in mono-energetic electrons (u_z = +-u): ionized fraction vs 1 - exp(-n sigma v xi), Lotz formula"
    echo "     implemented independently in Python:"
    local u
    for u in 0.010836 0.0198 0.06259; do
        $QS $E ea.u_drift="0 0 $u" eb.u_drift="0 0 -$u" output.dir=out_eimp_$u > /dev/null
        python3 eimpact_check.py rate out_eimp_$u 1e16 $u 0.0005 0.0005 Xe:1e-4:0
    done
    $QS $E xe.z0=1 output.dir=out_eimp_z1 > /dev/null
    echo " (b) Xe+ -> Xe2+ (high-charge Lotz form, 5p^5):"
    python3 eimpact_check.py rate out_eimp_z1 1e16 0.0198 0.0005 0.0005 Xe:1e-4:1
    $QS $E xe.eimpact.table_0=eimpact_table.txt output.dir=out_eimp_tab > /dev/null
    echo " (c) cross-section table (eimpact_table.txt):"
    python3 eimpact_check.py rate out_eimp_tab 1e16 0.0198 0.0005 0.0005 Xe:1e-4:0:eimpact_table.txt
    local L="eimpact.in xe.density=1 ea.density=1e-5 eb.density=1e-5 sec.tail_dump=1"
    $QS $L xe.ionization.nsplit=$(q 10000000 2000000) ionization.electron_loss=continuous output.dir=out_eimp_loss > /dev/null
    echo " (d) dense Xe (1e16 cm^-3), dilute 100 eV electrons, continuous loss with excitation: energy vs RK4 of"
    echo "     dE/dxi = -n [sigma (Ip + <W>) + sum_j sigma_j W_j] |u|/Delta:"
    python3 eimpact_check.py loss out_eimp_loss 1e16 0.0198 Xe:1:0:exc ea eb 2> /dev/null
    $QS $L xe.ionization.nsplit=10000000 xe.excitation=0 ea.ppc=$(q 256 64) eb.ppc=$(q 256 64) output.dir=out_eimp_disc > /dev/null
    echo " (d2) discrete collisions (default), no excitation: mean energy vs RK4 (straggling: second order), energy lost by"
    echo "     all electrons vs sum of (Ip + W) over the electron-impact births:"
    python3 eimpact_check.py loss out_eimp_disc 1e16 0.0198 Xe:1:0 ea eb 2> /dev/null
    export OMP_NUM_THREADS=1
    local M="eimpact.in xe.density=1 ea.density=1e-5 eb.density=1e-5 xe.ionization.nsplit=1000000 xe.ionization.born_dump=0"
    M="$M ea.tail_dump=0 eb.tail_dump=0 xi.max=200 time.steps=4 output.every=2"
    $QS $M checkpoint.every=2 checkpoint.keep=0 checkpoint.dir=out_eimp_chk output.dir=out_eimp_A > /dev/null
    $QS $M restart.from=out_eimp_chk/chk_000002 output.dir=out_eimp_B > /dev/null
    echo " (e) restart (expected: bit-identical):"
    python3 restart_check.py out_eimp_A out_eimp_B 0 || true
    if [ -n "$MPIRUN" ]; then
        $MPIRUN 2 $QS $M output.dir=out_eimp_P > /dev/null
        echo " 2 ranks vs serial (expected: bit-identical):"
        python3 cmp_runs.py out_eimp_A out_eimp_P 1e-14 || true
    fi
    M="$M xe.ionization.quiet=1"
    $QS $M checkpoint.every=2 checkpoint.keep=0 checkpoint.dir=out_eimp_chkq output.dir=out_eimp_qA > /dev/null
    $QS $M restart.from=out_eimp_chkq/chk_000002 output.dir=out_eimp_qB > /dev/null
    echo " (f) with xe.ionization.quiet = 1: restart (expected: bit-identical):"
    python3 restart_check.py out_eimp_qA out_eimp_qB 0 || true
    if [ -n "$MPIRUN" ]; then
        $MPIRUN 2 $QS $M output.dir=out_eimp_qP > /dev/null
        echo " 2 ranks vs serial (expected: bit-identical):"
        python3 cmp_runs.py out_eimp_qA out_eimp_qP 1e-14 || true
    fi
    unset OMP_NUM_THREADS
    rm -rf out_eimp*
}

sec24() {
    echo "== 24. 0D electron cascade in Xe: ionization + excitation (W-value) vs an independent event-by-event Monte Carlo =="
    local C="eimpact.in xe.density=1000 ea.density=1e-6 eb.density=1e-6 ea.ppc=64 eb.ppc=64 xe.ppc=16 xi.step=0.1 sec.tail_dump=1"
    C="$C xe.ionization.nsplit=2000000000"
    $QS $C xi.max=300 output.dir=out_casc_100 > /dev/null
    $QS $C xi.max=300 xe.excitation=0 output.dir=out_casc_100n > /dev/null
    echo " (a) 100 eV primaries (cascade_mc.py: same cross-sections, discrete collisions; measured W of Xe: 22.0 eV at high energy):"
    python3 eimpact_check.py cascade out_casc_100 100.16
    python3 eimpact_check.py cascade out_casc_100n 100.16 noexc
    if [ "$QUICK" = 0 ]; then
        $QS $C xi.max=1000 ea.ppc=4 eb.ppc=4 xe.ppc=16 ea.u_drift="0 0 0.06259" eb.u_drift="0 0 -0.06259" output.dir=out_casc_1k > /dev/null
        echo " (b) 1 keV primaries:"
        python3 eimpact_check.py cascade out_casc_1k 999.94
    fi
    rm -rf out_casc*
}

sec25() {
    echo "== 25. beam shot noise: noise power of the macro-particle beam vs the real protons (Poisson), before and after 1 m of drift =="
    $QS shot_noise.in output.dir=out_shot_lat > /dev/null
    python3 shot_check.py out_shot_lat "quiet lattice (default loader)"
    $QS shot_noise.in p.rods=8 p.dr=0.5 p.shot_noise=1 output.dir=out_shot_rods > /dev/null
    python3 shot_check.py out_shot_rods "rods + shot noise (rods 8, dr 0.5, shot_noise 1)"
    if [ -n "$MPIRUN" ]; then
        export OMP_NUM_THREADS=1
        local M="shot_noise.in p.rods=8 p.dr=0.5 p.shot_noise=1 time.steps=2 output.every=1 output.beam_every=0 beams.xi_shape=ngp"
        $QS $M output.dir=out_shot_s > /dev/null
        $MPIRUN 2 $QS $M output.dir=out_shot_p > /dev/null
        echo " 2 ranks vs serial (expected: bit-identical):"
        python3 cmp_runs.py out_shot_s out_shot_p 1e-14 || true
    fi
    rm -rf out_shot*
}

MODE=$(q full quick); TMP=$(mktemp -d); T0=$(date +%s)
run_section() {   # runs one section in its own subshell (environment changes stay local)
    local n=$1 t=$(date +%s)
    # results = stdout; stderr (Kokkos/MPI warnings, error messages) is kept apart and shown on failure
    ( set -e; "sec$n" ) > "$TMP/$n.out" 2> "$TMP/$n.err"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo " SECTION $n ABORTED (exit code $rc); last messages:" >> "$TMP/$n.out"
        tail -5 "$TMP/$n.err" | sed 's/^/   | /' >> "$TMP/$n.out"
    fi
    echo $(( $(date +%s) - t )) > "$TMP/$n.time"
}
running=0
for n in $SECTIONS; do
    run_section "$n" &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
done
wait
for n in $SECTIONS; do grep -v "^$" "$TMP/$n.out"; done > "$TMP/all.txt"
cat "$TMP/all.txt"
echo "-- timing ($MODE mode, $JOBS parallel, $OMP_NUM_THREADS threads each): $(for n in $SECTIONS; do printf "%s:%ss " $n $(cat $TMP/$n.time); done)total $(( $(date +%s) - T0 )) s" >&2
REF=reference_$MODE.txt; RC=0
if [ "$CMP" = 1 ]; then python3 compare_results.py "$REF" "$TMP/all.txt" $SECTIONS || RC=1; fi
if [ "$UPD" = 1 ]; then python3 compare_results.py --update "$REF" "$TMP/all.txt" $SECTIONS; fi
rm -rf "$TMP"
exit $RC
