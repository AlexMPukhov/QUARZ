#!/bin/sh
# Agent skills (skills/*/SKILL.md, docs/agent_skill.md) against the code, so that they cannot drift from it:
#   1. format of the open Agent Skills specification (agentskills.io): frontmatter name = directory name
#      (lower case, digits, hyphens, <= 64), description 1 ... 1024 characters, SKILL.md <= 500 lines; every file
#      of the skill that a SKILL.md names (references/, scripts/, examples/) exists
#   2. the copies of README.md sections 5 and 6 and of the example inputs equal their sources
#      (tools/sync_skills.sh --check)
#   3. every input key the hand-written skill files name (grid.rmax, <beam>.sigma_r, ...) is a key the code reads
#      (string literal in src/*.cpp) and is documented in README.md section 5
#   4. every ```quarz snippet of the skill files passes the input check (scripts/check_input.sh: a zero-step
#      solve, failing on errors and on keys that were not read); the check itself must catch a misspelt key
#   5. every input shipped in a skill passes the check and runs 2 steps with the real executable (finite numbers
#      in beams.txt and in the axis files)
#   6. the helper scripts: quarz_numbers.py against the beam charge the code reports
# usage: sh tests/agent_skill.sh <quarz>
#
# The same test exists in GEM-PIC-RZ (tests/agent_skill.sh there); the two differ only in the block
# "code-specific settings" and in the checks of the helper scripts.
abspath() { (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")"); }
EXE=$(abspath "$1")
SRC=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
export OMP_PROC_BIND=false
fail=0
bad() { echo "  FAILED: $*"; fail=1; }
# ---- code-specific settings -----------------------------------------------------------------------------------
SKILL="$SRC/skills/quarz-simulation"
# fixed key prefixes of the input; <beam>.key and <species>.key are user-named and checked by their suffix
PREFIXES="grid xi time pusher plasma laser output units checkpoint restart solver beams boundary scattering ionization diag"
PARSER="$SRC/src"                     # the sources that read the keys: "prefix.key" or ".key" as a string literal
REFERENCE="$SRC/README.md"            # the input reference: section 5 must name the key
MINSKILLS=2; MINKEYS=30; MININPUTS=4  # fewer found = the extraction is broken
check_input() { sh "$SKILL/scripts/check_input.sh" "$EXE" "$@"; }       # <input> [overrides]
# <input> <output dir>: 2 steps; an input with time.steps = 0 (one quasi-static solve) runs as it is
short_run() {
  inp=$1; out=$2; shift 2
  if awk -F= '/^[ \t]*time.steps[ \t]*=/ { v = $2 + 0 } END { exit !(v == 0) }' "$inp"; then
    "$EXE" "$inp" output.every=1 output.dir="$out" "$@"
  else
    "$EXE" "$inp" time.steps=2 output.every=1 output.dir="$out" "$@"
  fi
}
# ---- end of the code-specific settings ------------------------------------------------------------------------
PY=""
python3 -c "import sys" 2>/dev/null && PY=python3

echo "1. format (agentskills.io specification)"
nskill=0
for dir in "$SRC"/skills/*/; do
  dir=${dir%/}; name=$(basename "$dir"); f="$dir/SKILL.md"; nskill=$((nskill + 1))
  [ -f "$f" ] || { bad "$name: no SKILL.md"; continue; }
  [ "$(head -1 "$f")" = "---" ] || bad "$name: SKILL.md must start with the YAML frontmatter (---)"
  n=$(awk '/^---$/ { c++; next } c == 1 && /^name:/ { sub(/^name:[ \t]*/, ""); print; exit }' "$f")
  d=$(awk '/^---$/ { c++; next } c == 1 && /^description:/ { sub(/^description:[ \t]*/, ""); print; exit }' "$f")
  [ "$n" = "$name" ] || bad "$name: frontmatter name '$n' differs from the directory name"
  echo "$n" | grep -Eq '^[a-z0-9]+(-[a-z0-9]+)*$' || bad "$name: name must be lower case letters, digits, single hyphens"
  [ ${#n} -le 64 ] || bad "$name: name longer than 64 characters"
  [ ${#d} -ge 1 ] && [ ${#d} -le 1024 ] || bad "$name: description has ${#d} characters (1 ... 1024)"
  l=$(wc -l < "$f"); [ "$l" -le 500 ] || bad "$name: SKILL.md has $l lines (<= 500)"
  for ref in $(grep -oE '(references|scripts|examples)/[A-Za-z0-9_.-]+\.[a-z0-9]+' "$f" | sort -u); do
    [ -f "$dir/$ref" ] || [ -f "$SKILL/$ref" ] || bad "$name: SKILL.md names $ref, which is not in the skills"
  done
  echo "  $name: name ok, description ${#d} characters, $l lines"
done
[ "$nskill" -ge $MINSKILLS ] || bad "expected at least $MINSKILLS skills in skills/, found $nskill"

echo "2. copies of repository files"
sh "$SRC/tools/sync_skills.sh" --check || fail=1

echo "3. input keys named in the skill files"
# hand-written files only (the generated references are the copies checked in 2.)
MD=$(ls "$SRC"/skills/*/SKILL.md "$SRC"/skills/*/references/*.md | grep -vE '/(input|output)_reference.md$')
alt=$(echo $PREFIXES | tr ' ' '|')
awk '/^## / { on = (index($0, "## 5. ") == 1); next } on' "$REFERENCE" > ref5.md
cat $MD | grep -oE "(^|[^A-Za-z0-9_.>])(($alt)|<beam>|<species>)\.[a-z_][a-z_0-9]*(\.[a-z_][a-z_0-9]*)?" |
  sed -E 's/^[^a-z<]*//; s/^<([a-z]+\.)/\1/' | grep -vE '\.(key|in|txt|bin|h5|md|sh|py|cpp|hpp|png)$' | sort -u > keys.txt
nkey=0
while IFS= read -r full; do
  nkey=$((nkey + 1))
  key=${full#*.}
  case "$full" in
    # the code builds many keys from a prefix: "<name>" + ".key", or "output." + "key"
    "<"*) grep -rqF "\".$key\"" "$PARSER" || grep -rqF "\"$key\"" "$PARSER" ||
            bad "$full: the code reads no key ending in \"$key\"" ;;
    *)    grep -rqF "\"$full\"" "$PARSER" || grep -rqF "\".$key\"" "$PARSER" || grep -rqF "\"$key\"" "$PARSER" ||
            bad "$full: the code reads no key \"$full\"" ;;
  esac
  grep -qF "$full" ref5.md || grep -qF "\`$key" ref5.md ||
    bad "$full: not documented in section 5 of README.md"
done < keys.txt
echo "  $nkey keys checked against src/ and README.md section 5"
[ "$nkey" -ge $MINKEYS ] || bad "only $nkey keys found in the skill files: the extraction is broken"

echo "4. input snippets of the skill files, and the check itself"
nsnip=0
for f in $MD; do
  awk -v base="snip_$(basename "$(dirname "$f")")_$(basename "$f" .md)" '
    /^```quarz[ \t]*$/ { n++; out = base "_" n ".in"; next }
    /^```/ { out = ""; next }
    out != "" { print > out }' "$f"
done
for s in snip_*.in; do
  [ -f "$s" ] || continue
  nsnip=$((nsnip + 1))
  if check_input "$s" > "$s.log" 2>&1; then echo "  $s: ok"
  else bad "$s does not pass the check:"; tail -3 "$s.log"; fi
  if check_input "$s" grid.rmaxx=3 > "$s.neg.log" 2>&1; then bad "check_input.sh accepts the unknown key grid.rmaxx"; fi
  if check_input "$s" grid.type=nonsense > "$s.neg2.log" 2>&1; then bad "check_input.sh accepts grid.type=nonsense"; fi
done
[ "$nsnip" -ge 1 ] || bad "no \`\`\`quarz snippet found in the skill files"

echo "5. inputs shipped in the skills: check and 2 steps"
ninp=0
for inp in "$SRC"/skills/*/examples/*.in; do
  b=$(basename "$inp" .in); ninp=$((ninp + 1))
  check_input "$inp" > "$b.check.log" 2>&1 || { bad "$b: check"; tail -3 "$b.check.log"; continue; }
  if ! short_run "$inp" "out_$b" > "$b.run.log" 2>&1; then
    bad "$b: the run stopped"; tail -5 "$b.run.log"; continue
  fi
  grep -q '^Total run time' "$b.run.log" || bad "$b: the run did not finish"
  grep -q '^WARNING: input key' "$b.run.log" && bad "$b: $(grep -c '^WARNING: input key' "$b.run.log") key(s) not used"
  [ -s "out_$b/beams.txt" ] || bad "$b: no beams.txt"
  ls "out_$b"/axis_*.txt > /dev/null 2>&1 || bad "$b: no axis files"
  if cat "out_$b"/beams.txt "out_$b"/axis_*.txt 2>/dev/null | grep -v '^#' | grep -qiE 'nan|inf'; then
    bad "$b: NaN or Inf in the output"
  fi
  echo "  $b: ok ($(grep '^Total run time' "$b.run.log"))"
done
[ "$ninp" -ge $MININPUTS ] || bad "expected at least $MININPUTS example inputs in the skills, found $ninp"

echo "6. helper scripts"
if [ -n "$PY" ]; then
  for s in "$SKILL"/scripts/*.py; do
    python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$s" || bad "$(basename "$s"): syntax"
  done
  # The snippet of SKILL.md has a Gaussian beam with n0 = 2, sigma_r = 0.3, sigma_xi = 1 (cut at 3 sigma). The
  # code reports its charge in units of n0 (c/omega_p)^3; quarz_numbers.py must give n0 = 2 back from the
  # particle number and the sizes that belong to this charge at 1e17 cm^-3.
  s=$(ls snip_quarz-simulation_SKILL_1.in 2>/dev/null)
  if [ -n "$s" ] && "$EXE" "$s" time.steps=0 output.dir=out_snip > snip.run.log 2>&1; then
    python3 - "$SKILL/scripts/quarz_numbers.py" out_snip/beams.txt << 'PYEOF' || bad "quarz_numbers.py against the code"
import math, subprocess, sys
charge = abs(float([l for l in open(sys.argv[2]) if not l.startswith("#")][0].split()[5]))   # electrons: < 0
ideal = 2.0 * (2 * math.pi) ** 1.5 * 0.3 ** 2 * 1.0           # n0 (2 pi)^(3/2) sigma_r^2 sigma_xi
if abs(charge / ideal - 1) > 0.03:                           # the 3 sigma cut takes 1.4 %
    sys.exit(f"  beam charge of the code {charge:.5g}, Gaussian integral {ideal:.5g}")
n0 = 1e17
kp_um = 5.3141e5 / math.sqrt(n0) * 1e4
N = ideal * n0 * (kp_um * 1e-4) ** 3
out = subprocess.run([sys.executable, sys.argv[1], "--density", str(n0), "--particles", repr(N),
                      "--sigma-r-um", repr(0.3 * kp_um), "--sigma-z-um", repr(1.0 * kp_um)],
                     capture_output=True, text=True).stdout
val = [l for l in out.splitlines() if "<beam>.n0 =" in l]
if not val or abs(float(val[0].split("=")[1].split()[0]) / 2.0 - 1) > 1e-3:
    sys.exit("  quarz_numbers.py does not return <beam>.n0 = 2:\n" + out)
print(f"  quarz_numbers.py: <beam>.n0 = {float(val[0].split('=')[1].split()[0]):.5g} (input 2), "
      f"code charge / Gaussian integral = {charge / ideal:.4f}  ok")
PYEOF
    if python3 -c "import numpy" 2>/dev/null; then
      python3 "$SKILL/scripts/quick_look.py" out_snip --density 1e17 > look.log 2>&1 || { bad "quick_look.py"; tail -3 look.log; }
      grep -q "driver" look.log || bad "quick_look.py: no beam line"
    fi
  else
    bad "the snippet of SKILL.md did not run"
  fi
else
  echo "  skipped (no python3)"
fi

if [ $fail = 0 ]; then echo "agent skill test passed"; else echo "agent skill test FAILED"; fi
exit $fail
