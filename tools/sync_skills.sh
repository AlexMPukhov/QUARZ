#!/bin/sh
# Agent skills (skills/, see docs/agent_skill.md): the parts that are copies of repository files.
#   sh tools/sync_skills.sh           regenerate the copies (run it after editing README.md sections 5, 6 or the examples)
#   sh tools/sync_skills.sh --check   exit 1 if a copy differs from its source (ctest agent_skill)
# A skill is installed without the rest of the repository, so it carries its own copy of the input reference
# (README.md section 5), of the output formats (section 6) and of some example inputs; the check keeps them
# identical to the sources.
SRC=$(cd "$(dirname "$0")/.." && pwd)
SK="$SRC/skills/quarz-simulation"
INPUTS="awake_seeded_smi blowout_basic pwfa_ion_motion lwfa_laser"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

section() {  # <number> <title of the copy>: README.md from "## <number>. " to the next "## "
  echo "<!-- Generated from README.md (section $1) by tools/sync_skills.sh. Do not edit this copy. -->"
  echo "# $2"
  echo
  echo "Section numbers (§) refer to the README.md of https://github.com/AlexMPukhov/QUARZ."
  awk -v n="$1" '/^## / { on = (index($0, "## " n ". ") == 1); next } on' "$SRC/README.md"
}
section 5 "QUARZ input reference" > "$TMP/input_reference.md"
section 6 "QUARZ output files" > "$TMP/output_reference.md"

rc=0
put() {  # <generated file> <destination>
  if [ "$MODE" = "--check" ]; then
    if cmp -s "$1" "$2"; then echo "  in sync: ${2#$SRC/}"
    else echo "  OUT OF SYNC: ${2#$SRC/}  (run: sh tools/sync_skills.sh)"; rc=1; fi
  else
    mkdir -p "$(dirname "$2")" && cp "$1" "$2" && echo "  written: ${2#$SRC/}"
  fi
}
MODE=$1
put "$TMP/input_reference.md" "$SK/references/input_reference.md"
put "$TMP/output_reference.md" "$SK/references/output_reference.md"
for d in $INPUTS; do put "$SRC/examples/$d.in" "$SK/examples/$d.in"; done
exit $rc
