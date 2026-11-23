"""Compare a validation run with the stored reference, section by section.

usage: python3 compare_results.py reference.txt new.txt [sections...]
       python3 compare_results.py --update reference.txt new.txt [sections...]

Lines are compared in order. Their text without the numbers must agree exactly
(this catches e.g. "<-- above tolerance" markers, skipped or aborted sections); only the
"(file col)" location of a maximum difference may change.
Every number must agree within
  * 1 unit in its last printed digit, or relative 1e-3 (round-off of OpenMP atomics can
    change the last digit of a printed result), or
  * both values below 1e-6: they measure round-off themselves (e.g. MPI vs serial,
    parsed vs built-in profiles), so only "still round-off" is required, except that an
    exact 0 in the reference (bit-identity) must stay exactly 0.
Lines ending with "[verdict]" print numbers that legitimately differ between builds (random draws that hash the
particle state; round-off of a GPU restart): only their text, including the OK / FAIL verdict that the check
computed against its own expectation or tolerance, is compared, not the numbers.
"""
import re
import sys

NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?%?")


def sections(path):
    out, cur = {}, None
    for line in open(path):
        line = line.rstrip("\n")
        m = re.match(r"== (\d+)\.", line)
        if m:
            cur = int(m.group(1))
            out[cur] = []
        if cur is not None and line.strip():
            out[cur].append(line)
    return out


def ulp(tok):
    t = tok.rstrip("%")
    mant, _, exp = t.lower().partition("e")
    dec = len(mant.split(".")[1]) if "." in mant else 0
    return 10.0 ** (-dec + (int(exp) if exp else 0))


def num_ok(a, b):
    x, y = float(a.rstrip("%")), float(b.rstrip("%"))
    if x == y:
        return True
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return x != 0.0
    return abs(x - y) <= max(1.0001 * max(ulp(a), ulp(b)), 1e-3 * max(abs(x), abs(y)))


LOC = re.compile(r"\([^()]*\.(?:txt|bin)[^()]*\)")   # "(file col)": where a round-off maximum sits


def compare_line(r, n):
    r, n = LOC.sub("(...)", r), LOC.sub("(...)", n)
    if NUM.sub("#", r) != NUM.sub("#", n):
        return "text differs"
    if r.rstrip().endswith("[verdict]"):
        return None
    for a, b in zip(NUM.findall(r), NUM.findall(n)):
        if not num_ok(a, b):
            return "%s -> %s" % (a, b)
    return None


def main():
    args = sys.argv[1:]
    update = args and args[0] == "--update"
    if update:
        args = args[1:]
    ref_path, new_path = args[0], args[1]
    new = sections(new_path)
    want = [int(s) for s in args[2:]] or sorted(new)
    try:
        ref = sections(ref_path)
    except FileNotFoundError:
        ref = {}
    if update:
        for s in want:
            if s in new:
                ref[s] = new[s]
        with open(ref_path, "w") as f:
            for s in sorted(ref):
                f.write("\n".join(ref[s]) + "\n")
        print("reference %s updated (sections %s)" % (ref_path, " ".join(map(str, want))))
        return 0
    bad = 0
    for s in want:
        if s not in ref:
            print("section %2d: NO REFERENCE" % s)
            bad += 1
            continue
        r, n = ref[s], new.get(s, [])
        probs = []
        if len(r) != len(n):
            probs.append("%d lines instead of %d" % (len(n), len(r)))
        for i, (a, b) in enumerate(zip(r, n)):
            p = compare_line(a, b)
            if p:
                probs.append("line %d: %s\n      ref: %s\n      new: %s" % (i + 1, p, a.strip()[:150], b.strip()[:150]))
        print("section %2d: %s" % (s, "PASS" if not probs else "FAIL"))
        for p in probs[:5]:
            print("    " + p)
        bad += bool(probs)
    print("validation: %s (%d of %d sections failed)" % ("PASS" if not bad else "FAIL", bad, len(want)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
