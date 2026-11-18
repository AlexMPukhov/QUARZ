#!/bin/sh
# Check a QUARZ input file without running it: one quasi-static solve (time.steps=0) into a scratch directory.
#
#   sh check_input.sh <path to quarz> <input file> [key=value ...]
#
# Prints the start-up summary of the code (grid, box, species, beams, output) and fails (exit 1) if
#   - the code stops with an error (wrong value, missing key, missing file), or
#   - a key of the input was never read: QUARZ only warns about these at the end of a run, but an unread key is
#     almost always a misspelt one, and the run then silently uses the default.
# Keys that act only during the time steps (checkpoints, adaptive time step) are read at start-up as well.
# The solve itself costs one sweep over the box: seconds for 10^4 slices.
if [ $# -lt 2 ]; then echo "usage: sh check_input.sh <quarz> <input file> [key=value ...]"; exit 2; fi
exe=$1; inp=$2; shift 2
tmp=$(mktemp -d "${TMPDIR:-/tmp}/quarz_check.XXXXXX") || exit 2
"$exe" "$inp" "$@" time.steps=0 output.dir="$tmp/out" > "$tmp/log" 2>&1
rc=$?
grep -v "^WARNING: input key" "$tmp/log"
bad=$(grep -c "^WARNING: input key" "$tmp/log")
err=$(grep -c "^ERROR" "$tmp/log")
grep "^WARNING: input key" "$tmp/log" | sed 's/^WARNING/UNKNOWN KEY/'
rm -rf "$tmp"
if [ "$rc" -ne 0 ] || [ "$err" -ne 0 ]; then echo "check_input: FAILED (error above)"; exit 1; fi
if [ "$bad" -ne 0 ]; then echo "check_input: FAILED ($bad key(s) not used: misspelt, or not valid in this setup)"; exit 1; fi
echo "check_input: ok"
