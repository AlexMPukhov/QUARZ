#!/usr/bin/env python3
"""Convert a QUARZ input deck from the v1.0 convention (xi = t - z, beam head at the
smallest xi) to the current one (xi = z - t, beam moving towards +xi, head at the largest xi).

    tools/convert_deck.py old.in > new.in          # or: -o new.in, or -i (in place)
    tools/convert_deck.py --beam-file old.txt new.txt   # profile = file: negate the xi column

What changes (every other line is copied unchanged, comments included):
    xi.min = a, xi.max = b              ->  xi.min = -b, xi.max = -a   (old default xi.min = 0)
    <x>.xi0, <beam>.xi_cut_head         ->  negated
    <beam>.x_slope, <beam>.y_slope      ->  negated (d x_c / d xi)
    <beam>.xi_range = a b               ->  -b -a
    output.beam_slices_range = a b      ->  -b -a
    beam functions of xi (density(x,y,xi), ux_mean, uy_mean, uz_mean, ux_std, uy_std, uz_std):
                                            the xi variable is replaced by (-xi), so the function
                                            describes the same beam
Plasma functions (x,y,z) use the lab position z and are not changed. The converted deck gives
the same physics; the numbers are negated exactly, so the results are identical to the old run.
"""
import argparse
import re
import sys

BEAM_FUNCS = ("density", "ux_mean", "uy_mean", "uz_mean", "ux_std", "uy_std", "uz_std")
PLAIN_FUNCS = ("ux_mean", "uy_mean", "uz_mean", "ux_std", "uy_std", "uz_std")   # default vars x, y, xi
MARK = "# converted to the xi = z - t convention (beam head at the largest xi) by tools/convert_deck.py"

LINE = re.compile(r'^(\s*)([A-Za-z_][A-Za-z0-9_.]*)(\([^)]*\))?(\s*=\s*)(.*?)$')


def neg(tok):
    """exact negation of a number written as text"""
    t = tok.strip()
    float(t)   # must be a number
    if t.startswith("-"):
        return t[1:]
    if t.startswith("+"):
        t = t[1:]
    return "0" if float(t) == 0.0 else "-" + t


def split_value(v):
    """value part, trailing comment (keeps quotes intact)"""
    q = None
    for i, c in enumerate(v):
        if c in "\"'":
            q = None if q == c else (c if q is None else q)
        elif c == "#" and q is None:
            return v[:i].rstrip(), v[len(v[:i].rstrip()):]
    return v.rstrip(), v[len(v.rstrip()):]


def flip_function(expr, var):
    return re.sub(r'(?<![A-Za-z0-9_.])' + re.escape(var) + r'(?![A-Za-z0-9_])', "(-" + var + ")", expr)


def convert(lines):
    keys = {}
    for ln in lines:
        m = LINE.match(ln)
        if m and not ln.lstrip().startswith("#"):
            keys[m.group(2) + (m.group(3) or "")] = split_value(m.group(5))[0]
    if any(MARK in ln for ln in lines):
        raise SystemExit("convert_deck: this deck is already converted")
    beams = keys.get("beams", "").strip('"').split()
    old_min = keys.get("xi.min", "0")
    old_max = keys.get("xi.max")
    out, warn = [MARK + "\n"], []
    for ln in lines:
        m = LINE.match(ln)
        if not m or ln.lstrip().startswith("#"):
            out.append(ln)
            continue
        ind, key, args, eq, rest = m.groups()
        val, com = split_value(rest)
        nl = "\n" if ln.endswith("\n") else ""
        com = com.rstrip("\n")
        new = None
        base, _, suf = key.rpartition(".")
        if key == "xi.min":
            new = neg(old_max) if old_max is not None else None
            if new is None:
                warn.append("xi.max missing")
        elif key == "xi.max":
            new = neg(old_min)
            if "xi.min" not in keys:   # old default xi.min = 0 -> the new xi.min is -xi.max
                out.append(f"{ind}xi.min{eq}{neg(val)}{nl}")
        elif suf in ("xi0", "xi_cut_head", "x_slope", "y_slope") and not args:
            new = neg(val)
        elif (suf == "xi_range" and base in beams) or key == "output.beam_slices_range":
            a = val.split()
            if len(a) == 2:
                new = f"{neg(a[1])} {neg(a[0])}"
        elif base in beams and suf in BEAM_FUNCS and args:
            vars_ = [s.strip() for s in args[1:-1].split(",")]
            if len(vars_) == 3:
                new = flip_function(val, vars_[2])
        elif base in beams and suf in PLAIN_FUNCS and not args:
            try:
                float(val.strip('"'))
            except ValueError:
                new = flip_function(val, "xi")
        if base in beams and suf == "profile" and val.strip('"') == "file":
            warn.append(f"{base}.profile = file: convert the beam file {keys.get(base + '.file', '?')} with --beam-file")
        out.append(ln if new is None else f"{ind}{key}{args or ''}{eq}{new}{com}{nl}")
    return out, warn


def convert_beam_file(src, dst):
    with open(src) as f, open(dst, "w") as g:
        g.write("# xi = z - t convention (converted by tools/convert_deck.py)\n")
        for ln in f:
            p = ln.split()
            if ln.lstrip().startswith("#") or len(p) < 7:
                g.write(ln)
                continue
            p[5] = neg(p[5])
            g.write(" ".join(p) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deck", nargs="?")
    ap.add_argument("-o", "--output")
    ap.add_argument("-i", "--in-place", action="store_true")
    ap.add_argument("--beam-file", nargs=2, metavar=("OLD", "NEW"))
    a = ap.parse_args()
    if a.beam_file:
        convert_beam_file(*a.beam_file)
        return
    if not a.deck:
        ap.error("give a deck")
    with open(a.deck) as f:
        lines = f.readlines()
    out, warn = convert(lines)
    for w in warn:
        print("convert_deck: WARNING: " + w, file=sys.stderr)
    dst = a.deck if a.in_place else a.output
    if dst:
        with open(dst, "w") as f:
            f.writelines(out)
    else:
        sys.stdout.writelines(out)


if __name__ == "__main__":
    main()
