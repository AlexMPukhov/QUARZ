#!/bin/sh
# The common input standard of the VLPL family in QUARZ (README "Input file"): blocks, ';', units, strict keys,
# --check. References: the same run written in the old flat form.
#   (1) block form (nested and named blocks, one-line blocks with ';') = flat form: beams.txt byte-identical
#   (2) the same deck with lengths in um, times in fs and densities in cm^-3 = the normalized deck with the same
#       reference density (1e-7 relative). The reference density is part of the physics: with it QUARZ switches on
#       the Coulomb scattering of the beams (random kicks: its part of the emittance is not reproducible to round-off
#       when the input changes in the last digit), so both runs get the density and scattering = 0
#   (3) a key given twice, an unknown key, a unit without units.n0, an unknown unit: error before the first step
#   (4) --check: the input is read and set up, no step is done
#   (5) the common key names (grid.m_max, grid.r_max; bunch: species, charge with a unit = total charge, energy,
#       sigma_x/y/z, zeta_center, macroparticles) = the old keys: the total charge and the kinetic energy are
#       computed here independently (awk) from the peak density and gamma of the flat deck (1e-7 relative)
#   (6) laser: wavelength, tau_fwhm, waist, z_focus = lambda0_um, L0, w0, focus: laser.txt of step 0 (1e-7);
#       two names of one quantity in one deck: error
#   sh tests/deck_standard.sh <quarz executable>
Q=$1
W=$(mktemp -d "${TMPDIR:-/tmp}/quarz_deck.XXXXXX")
trap 'rm -rf "$W"' EXIT
cd "$W" || exit 1
export OMP_PROC_BIND=false
fail=0
ok() { echo "  [PASS] $1"; }
bad() { echo "  [FAIL] $1"; fail=1; }

cat > flat.in <<'IN'
xi.convention = z-t
grid.type = regions
grid.rmax = 8
grid.regions = 8:0.05
xi.max = 0
xi.min = -12
xi.step = 0.05
time.dt = 5
time.steps = 2
plasma.species = electrons
plasma.rmax = 7
electrons.ppc = 2
beams = driver witness
driver.n0 = 0.5
driver.sigma_r = 0.5
driver.sigma_xi = 1
driver.xi0 = -4
driver.gamma = 2000
driver.nparticles = 20000
witness.n0 = 0.05
witness.sigma_r = 0.3
witness.sigma_xi = 0.3
witness.xi0 = -8.5
witness.gamma = 400
witness.nparticles = 5000
output.every = 0
IN

# ---- (1) the same in blocks
cat > block.in <<'IN'
xi { convention = z-t; max = 0; min = -12; step = 0.05 }
grid {
  type = regions      # a comment
  rmax = 8
  regions = 8:0.05
}
time { dt = 5; steps = 2 }
plasma { rmax = 7 }
species electrons { ppc = 2 }
bunch driver {
  n0 = 0.5; sigma_r = 0.5; sigma_xi = 1
  xi0 = -4
  gamma = 2000
  nparticles = 20000
}
bunch witness { n0 = 0.05; sigma_r = 0.3; sigma_xi = 0.3; xi0 = -8.5; gamma = 400; nparticles = 5000; scattering = 0 }
output { every = 0 }
IN
"$Q" flat.in output.dir=flat > flat.log 2>&1 || { bad "the flat deck runs"; cat flat.log; }
"$Q" block.in output.dir=block > block.log 2>&1 || { bad "the block deck runs"; cat block.log; }
if [ -s flat/beams.txt ] && cmp -s flat/beams.txt block/beams.txt; then ok "block form = flat form (beams.txt identical, $(wc -l < flat/beams.txt) lines)"
else bad "block form = flat form (beams.txt)"; fi

# ---- (2) with units: n0 = 1e17 cm^-3
KP=$(awk 'BEGIN { printf "%.17g", 2.99792458e8 / (5.64146e4 * sqrt(1e17)) * 1e6 }')   # 1/k_p in um
u() { awk -v x="$1" -v k="$KP" 'BEGIN { printf "%.17g", x * k }'; }                     # normalized length -> um
t() { awk -v x="$1" 'BEGIN { printf "%.17g", x / (5.64146e4 * sqrt(1e17)) * 1e15 }'; }  # normalized time -> fs
cat > units.in <<IN
units { n0 = 1e17 cm^-3 }
xi { convention = z-t; max = 0 um; min = $(u -12) um; step = $(u 0.05)um }
grid { type = regions; rmax = $(u 8) um; regions = $(u 8)um:$(u 0.05)um }
time { dt = $(t 5) fs; steps = 2 }
plasma { rmax = $(u 7) um }
species electrons { ppc = 2 }
bunch driver { n0 = 0.5e17 cm^-3; sigma_r = $(u 0.5) um; sigma_xi = $(u 1) um; xi0 = $(u -4) um; gamma = 2000; nparticles = 20000; scattering = 0 }
bunch witness { n0 = 5e15cm^-3; sigma_r = $(u 0.3) um; sigma_xi = $(u 0.3) um; xi0 = $(u -8.5) um; gamma = 400; nparticles = 5000; scattering = 0 }
output { every = 0 }
IN
"$Q" flat.in units.n0_cm3=1e17 driver.scattering=0 witness.scattering=0 output.dir=flatn0 > flatn0.log 2>&1 || { bad "the flat deck with units.n0_cm3 runs"; cat flatn0.log; }
"$Q" units.in output.dir=units > units.log 2>&1 || { bad "the deck with units runs"; cat units.log; }
d=$(awk 'NR == FNR { for (i = 1; i <= NF; i++) a[FNR, i] = $i; n[FNR] = NF; next }
         /^#/ { next }
         { for (i = 1; i <= NF; i++) { x = a[FNR, i] + 0; y = $i + 0; e = x - y; if (e < 0) e = -e; s = (x < 0 ? -x : x) + 1e-30;
                                       if ($i ~ /^[-+0-9.eE]+$/ && e / s > m && s > 1e-12) m = e / s } }
         END { printf "%.1e", m }' flatn0/beams.txt units/beams.txt)
if [ -s units/beams.txt ] && awk -v d="$d" 'BEGIN { exit !(d < 1e-7) }'; then ok "deck with units = normalized deck (max relative difference $d, 1/k_p = $KP um)"
else bad "deck with units = normalized deck (max relative difference $d)"; head -n 4 flatn0/beams.txt units/beams.txt | cut -c1-400; diff flatn0.log units.log | head -n 20; fi

# ---- (3) errors before the first step
expect() {   # <name> <pattern> <deck> [args]
    n=$1; p=$2; f=$3; shift 3
    "$Q" "$f" "$@" output.dir=err > err.log 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -q "$p" err.log && ! grep -q "^step " err.log; then ok "$n: $(grep -m1 "$p" err.log | cut -c1-110)"
    else bad "$n (exit $rc)"; tail -n 3 err.log; fi
}
{ cat flat.in; echo "driver.gamma = 3000"; } > twice.in
expect "a key given twice is an error" "given twice" twice.in
{ cat flat.in; echo "driver.gama = 3000"; } > typo.in
expect "an unknown key is an error before the first step" "unknown input key" typo.in
sed 's/^xi.step = 0.05/xi.step = 0.84 um/' flat.in > nounit.in
expect "a unit without the reference density is an error" "needs the reference" nounit.in
{ echo "units.n0 = 1e17 cm^-3"; sed 's/^xi.step = 0.05/xi.step = 0.84 parsec/' flat.in; } > badunit.in
expect "an unknown unit is an error" "unknown unit" badunit.in
{ cat flat.in; echo "driver.espred = 0.01"; } > typo2.in
expect "a misspelt key of a bunch is an error" "unknown input key" typo2.in
# a known key that this setup does not read (the file of a Gaussian bunch): a warning before the first step, the run goes on
{ cat flat.in; echo "driver.file = nothing.bin"; } > unused.in
"$Q" unused.in output.dir=unused > unused.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q "^WARNING: input key 'driver.file' was not used" unused.log && cmp -s flat/beams.txt unused/beams.txt &&
   [ "$(grep -n "^WARNING: input key" unused.log | cut -d: -f1 | head -n 1)" -lt "$(grep -n "^step " unused.log | cut -d: -f1 | head -n 1)" ]; then
    ok "a known key that the setup does not read: warning before the first step, same result"
else bad "a known key that the setup does not read (exit $rc)"; tail -n 4 unused.log; fi
printf 'xi { step = 0.05\n' > open.in
expect "an unclosed block is an error" "not closed" open.in

# ---- (4) --check
"$Q" flat.in --check output.dir=chk > chk.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q "^check: input ok" chk.log && ! grep -q "^step " chk.log; then ok "--check: $(grep '^check:' chk.log)"
else bad "--check (exit $rc)"; tail -n 3 chk.log; fi
"$Q" typo.in --check output.dir=chk > chk.log 2>&1 && bad "--check accepts a misspelt key" || ok "--check refuses a misspelt key"

# max relative difference of the numbers of two text files (lines starting with # skipped)
maxdiff() {
    awk 'NR == FNR { for (i = 1; i <= NF; i++) a[FNR, i] = $i; next }
         /^#/ { next }
         { for (i = 1; i <= NF; i++) { x = a[FNR, i] + 0; y = $i + 0; e = x - y; if (e < 0) e = -e; s = (x < 0 ? -x : x) + 1e-30;
                                       if ($i ~ /^[-+0-9.eE]+$/ && e / s > m && s > 1e-12) m = e / s } }
         END { printf "%.1e", m }' "$1" "$2"
}

# ---- (5) common key names against the flat deck (reference density 1e17 cm^-3, no scattering)
# total charge of a Gaussian bunch cut at 3 sigma: Q = e n_b (2 pi)^(3/2) sigma_r^2 sigma_z (1 - exp(-9/2)) erf(3/sqrt 2)
qpc() { awk -v nb="$1" -v sr="$2" -v sz="$3" -v k="$KP" 'BEGIN { pi = 3.14159265358979323846;
    printf "%.17g", 1.602176634e-19 * nb * 1e17 * (2 * pi) ^ 1.5 * sr * sr * sz * (k * 1e-4) ^ 3 * (1 - exp(-4.5)) * 0.99730020393673981 * 1e12 }'; }
mev() { awk -v g="$1" 'BEGIN { printf "%.17g", (g - 1) * 0.51099895 }'; }
cat > common.in <<IN
units { n0 = 1e17 cm^-3 }
xi { convention = z-t; max = 0; min = -12; step = 0.05 }
grid { type = regions; r_max = 8; regions = 8:0.05; m_max = 0 }
time { dt = 5; steps = 2 }
plasma { rmax = 7 }
species electrons { ppc = 2 }
bunch driver {
  species = electron
  charge = $(qpc 0.5 0.5 1) pC
  energy = $(mev 2000) MeV
  sigma_x = $(u 0.5) um; sigma_y = $(u 0.5) um; sigma_z = $(u 1) um
  zeta_center = $(u 8) um            # 8 behind... from the rear end: xi0 = xi.min + 8 = -4
  macroparticles = 20000
  scattering = 0
}
bunch witness { n0 = 0.05; sigma_r = 0.3; sigma_xi = 0.3; xi0 = -8.5; gamma = 400; nparticles = 5000; scattering = 0 }
output { every = 0 }
IN
"$Q" common.in output.dir=common > common.log 2>&1 || { bad "the deck with the common names runs"; tail -n 5 common.log; }
d=$(maxdiff flatn0/beams.txt common/beams.txt)
if [ -s common/beams.txt ] && awk -v d="$d" 'BEGIN { exit !(d < 1e-7) }'; then
    ok "common names (species, total charge $(qpc 0.5 0.5 1) pC, energy $(mev 2000) MeV, sigma_x/y/z, zeta_center) = old keys (max relative difference $d)"
else bad "common names = old keys (max relative difference $d)"; head -n 3 flatn0/beams.txt common/beams.txt | cut -c1-300; fi

# ---- (6) laser names; two names of one quantity
cat > laser_old.in <<'IN'
xi.convention = z-t
units.n0_cm3 = 1e18
grid.type = regions
grid.rmax = 14
grid.regions = 1:0.02 5:0.05 14:0.2
xi.min = -16
xi.max = 0
xi.step = 0.02
time.dt = 20
time.steps = 0
plasma.species = electrons
electrons.ppc = 2
laser.lambda0_um = 0.8
laser.a0 = 2.0
laser.w0 = 4.0
laser.L0 = 2.3
laser.xi0 = -7.0
laser.focus = 30
output.every = 0
IN
KP18=$(awk 'BEGIN { printf "%.17g", 2.99792458e8 / (5.64146e4 * sqrt(1e18)) * 1e6 }')     # 1/k_p in um at 1e18 cm^-3
sed -e '/^laser.lambda0_um/d' -e '/^laser.w0/d' -e '/^laser.L0/d' -e '/^laser.focus/d' laser_old.in > laser_new.in
cat >> laser_new.in <<IN
laser.wavelength = 0.8 um
laser.waist = $(awk -v k="$KP18" 'BEGIN { printf "%.17g", 4.0 * k }') um
laser.tau_fwhm = $(awk 'BEGIN { printf "%.17g", 2.3 * sqrt(2 * log(2)) / (5.64146e4 * sqrt(1e18)) * 1e15 }') fs
laser.z_focus = $(awk -v k="$KP18" 'BEGIN { printf "%.17g", 30 * k * 1e-3 }') mm
IN
"$Q" laser_old.in output.dir=lold > lold.log 2>&1 || { bad "the laser deck runs"; tail -n 5 lold.log; }
"$Q" laser_new.in output.dir=lnew > lnew.log 2>&1 || { bad "the laser deck with the common names runs"; tail -n 5 lnew.log; }
d=$(maxdiff lold/laser.txt lnew/laser.txt)
if [ -s lnew/laser.txt ] && awk -v d="$d" 'BEGIN { exit !(d < 1e-7) }'; then
    ok "laser: wavelength, tau_fwhm, waist, z_focus = lambda0_um, L0, w0, focus (laser.txt, max relative difference $d)"
else bad "laser names (max relative difference $d)"; head -n 3 lold/laser.txt lnew/laser.txt | cut -c1-300; fi
{ cat laser_new.in; echo "laser.L0 = 2.3"; } > two.in
expect "two names of one quantity (laser.tau_fwhm and laser.L0) are an error" "two names of one quantity" two.in
sed 's/sigma_y = [^;]*;/sigma_y = 1 um;/' common.in > oval.in
expect "sigma_x differing from sigma_y is an error (round beams)" "are round" oval.in

[ $fail -eq 0 ] && echo "deck_standard: all passed" || echo "deck_standard: FAILED"
exit $fail
