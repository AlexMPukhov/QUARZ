#!/bin/sh
# The common input standard of the VLPL family in QUARZ (README "Input file"): blocks, ';', units, strict keys,
# --check. References: the same run written in the old flat form.
#   (1) block form (nested and named blocks, one-line blocks with ';') = flat form: beams.txt byte-identical
#   (2) the same deck with lengths in um, times in fs and densities in cm^-3 = the normalized deck (1e-9 relative)
#   (3) a key given twice, an unknown key, a unit without units.n0, an unknown unit: error before the first step
#   (4) --check: the input is read and set up, no step is done
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
bunch witness { n0 = 0.05; sigma_r = 0.3; sigma_xi = 0.3; xi0 = -8.5; gamma = 400; nparticles = 5000 }
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
bunch driver { n0 = 0.5e17 cm^-3; sigma_r = $(u 0.5) um; sigma_xi = $(u 1) um; xi0 = $(u -4) um; gamma = 2000; nparticles = 20000 }
bunch witness { n0 = 5e15cm^-3; sigma_r = $(u 0.3) um; sigma_xi = $(u 0.3) um; xi0 = $(u -8.5) um; gamma = 400; nparticles = 5000 }
output { every = 0 }
IN
"$Q" units.in output.dir=units > units.log 2>&1 || { bad "the deck with units runs"; cat units.log; }
d=$(awk 'NR == FNR { for (i = 1; i <= NF; i++) a[FNR, i] = $i; n[FNR] = NF; next }
         /^#/ { next }
         { for (i = 1; i <= NF; i++) { x = a[FNR, i] + 0; y = $i + 0; e = x - y; if (e < 0) e = -e; s = (x < 0 ? -x : x) + 1e-30;
                                       if ($i ~ /^[-+0-9.eE]+$/ && e / s > m && s > 1e-12) m = e / s } }
         END { printf "%.1e", m }' flat/beams.txt units/beams.txt)
if [ -s units/beams.txt ] && awk -v d="$d" 'BEGIN { exit !(d < 1e-9) }'; then ok "deck with units = normalized deck (max relative difference $d, 1/k_p = $KP um)"
else bad "deck with units = normalized deck (max relative difference $d)"; fi

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
printf 'xi { step = 0.05\n' > open.in
expect "an unclosed block is an error" "not closed" open.in

# ---- (4) --check
"$Q" flat.in --check output.dir=chk > chk.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q "^check: input ok" chk.log && ! grep -q "^step " chk.log; then ok "--check: $(grep '^check:' chk.log)"
else bad "--check (exit $rc)"; tail -n 3 chk.log; fi
"$Q" typo.in --check output.dir=chk > chk.log 2>&1 && bad "--check accepts a misspelt key" || ok "--check refuses a misspelt key"

[ $fail -eq 0 ] && echo "deck_standard: all passed" || echo "deck_standard: FAILED"
exit $fail
