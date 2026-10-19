"""Section 17: the GUI server (tools/quarz_gui.py) without a browser.
Starts the server, runs a small deck through /api/run, waits for the end and checks that
(1) the run finished and its live frames ('gui' diagnostic group) exist for every requested step,
(2) /api/lines (on-axis cut) equals the field file read directly, (3) /api/map is finite with the
requested size, (4) /api/beams has the steps of beams.txt, (5) requests without the token fail.
Remote machines (only where available here): (6) a Slurm batch job (sbatch), (7) ssh to localhost on
the shared file system, (8) ssh with a separate remote directory (deck copied, output pulled by rsync);
each must give the same frames as the local run, bit for bit. These need a Slurm installation and
ssh key login to localhost, so they run only with --remote (run_validation.sh: QUARZ_TEST_REMOTE=1)
and are not part of the stored references.
usage: gui_check.py deck exe port [--remote]"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_fields  # noqa: E402

deck, exe, port = os.path.abspath(sys.argv[1]), sys.argv[2], int(sys.argv[3])
remote = "--remote" in sys.argv[4:]
tok = "check"
gui = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "quarz_gui.py")
import shutil  # noqa: E402

exe = os.path.abspath(exe)
here = os.getcwd()


def have_slurm():
    if not shutil.which("sbatch"):
        return False
    r = subprocess.run(["sinfo", "-h", "-o", "%a"], capture_output=True, text=True)
    return r.returncode == 0 and "up" in r.stdout


def have_ssh():
    if not (shutil.which("ssh") and shutil.which("rsync")):
        return False
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "true"],
                       capture_output=True, text=True)
    return r.returncode == 0


machines = {}
if remote and have_slurm():
    machines["testslurm"] = {"type": "slurm", "exe": exe, "time": "00:05:00", "env": {"OMP_PROC_BIND": "false"},
                             # test cluster without PMI: mpirun inside the allocation instead of srun
                             "launcher": "mpirun -np {ranks}" + (" --allow-run-as-root" if os.geteuid() == 0 else "")}
if remote and have_ssh():
    machines["testssh"] = {"type": "ssh", "host": "localhost", "exe": exe, "env": {"OMP_PROC_BIND": "false"}}
    machines["testrsync"] = {"type": "ssh", "host": "localhost", "exe": exe, "env": {"OMP_PROC_BIND": "false"},
                             "remote_dir": os.path.join(here, "out_gui_remote")}
mfile = os.path.join(here, "out_gui_machines.json")
with open(mfile, "w") as f:
    json.dump({"machines": machines}, f)
srv = subprocess.Popen([sys.executable, gui, "--no-browser", "--port", str(port), "--token", tok, "--exe", exe,
                        "--machines", mfile],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
base = f"http://127.0.0.1:{port}"


def call(path, params=None, body=None, token=tok):
    q = "?" + "&".join(f"{k}={v}" for k, v in (params or {}).items()) if params else ""
    req = urllib.request.Request(base + path + q, data=None if body is None else json.dumps(body).encode(),
                                 headers={"X-Token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


try:
    for _ in range(50):
        try:
            call("/api/status")
            break
        except Exception:
            time.sleep(0.2)
    text = open(deck).read()
    work = os.path.abspath("out_gui_deck.in")
    call("/api/run", body={"path": work, "text": text, "threads": 1, "ranks": 1, "every": 2, "rmax": 1.0,
                           "xi_stride": 2, "fields": "ez psi ne", "extra": "output.dir=out_gui\ntime.steps=6\noutput.every=0"})
    t0 = time.time()
    while call("/api/status")["running"] and time.time() - t0 < 300:
        time.sleep(0.5)
    st = call("/api/status")
    fr = call("/api/frames")
    ok1 = st["rc"] == 0 and fr["frames"] == [0, 2, 4, 6] and sorted(fr["comps"]) == ["ez", "ne", "psi"]
    print(f"  run finished (exit {st['rc']}), frames {fr['frames']}, fields {fr['comps']}  {'OK' if ok1 else 'FAIL'}")
    d = read_fields(os.path.join("out_gui", "gui", "fields_000004.bin"))
    L = call("/api/lines", {"step": 4, "comps": "ez,psi"})
    e2 = max(float(np.max(np.abs(np.array(L["series"][c]) - d[c][:, 0]))) for c in ("ez", "psi"))
    print(f"  on-axis cut vs the field file: max difference {e2:.1e}, r_max of the frame {d['r'][-1]:.3f}, "
          f"{len(d['xi'])} of the box slices  {'OK' if e2 == 0 else 'FAIL'}")
    M = call("/api/map", {"step": 4, "comp": "ne", "nr": 50, "nxi": 100})
    import base64
    a = np.frombuffer(base64.b64decode(M["data"]), dtype=np.float32)
    ok3 = a.size == M["nxi"] * M["nr"] and M["nr"] == 50 and M["nxi"] <= 100 and np.all(np.isfinite(a))
    print(f"  2D map: {M['nxi']} x {M['nr']} values, finite: {bool(np.all(np.isfinite(a)))}  {'OK' if ok3 else 'FAIL'}")
    B = call("/api/beams")
    ok4 = all(len(v["t"]) == 7 for v in B.values()) and len(B) > 0
    print(f"  beams: {', '.join(f'{k} {len(v[chr(116)])} steps' for k, v in B.items())}  {'OK' if ok4 else 'FAIL'}")
    try:
        call("/api/status", token="wrong")
        code = 200
    except urllib.error.HTTPError as e:
        code = e.code
    print(f"  request without the token: HTTP {code}  {'OK' if code == 403 else 'FAIL'}")
    ref = {s: read_fields(os.path.join("out_gui", "gui", f"fields_{s:06d}.bin")) for s in (0, 2, 4, 6)}
    names = [m["name"] for m in call("/api/init")["machines"]]
    for mach, label in (("testslurm", "Slurm batch job"), ("testssh", "ssh, shared file system"),
                        ("testrsync", "ssh, remote directory + rsync")):
        if not remote:
            break
        if mach not in machines:
            print(f"  {label}: not available here, skipped")
            continue
        od = "out_gui_" + mach
        body = {"path": os.path.abspath(f"out_gui_{mach}.in"), "text": text, "threads": 1, "ranks": 1, "every": 2,
                "rmax": 1.0, "xi_stride": 2, "fields": "ez psi ne", "machine": mach, "time": "00:05:00",
                "extra": f"output.dir={od}\ntime.steps=6\noutput.every=0"}
        scr = call("/api/script", body=body)["script"]
        call("/api/run", body=body)
        t0 = time.time()
        while call("/api/status")["running"] and time.time() - t0 < 300:
            time.sleep(0.5)
        st = call("/api/status")
        fr = call("/api/frames")
        diff = 0.0
        for s_ in (0, 2, 4, 6):
            fn = os.path.join(od, "gui", f"fields_{s_:06d}.bin")
            if not os.path.exists(fn):
                diff = float("inf")
                break
            d2 = read_fields(fn)
            diff = max([diff] + [float(np.max(np.abs(d2[c] - ref[s_][c]))) for c in ("ez", "psi", "ne")])
        extra_ok = ("#SBATCH --ntasks=1" in scr and "jobid" in st) if mach == "testslurm" else ("cd " in scr)
        ok = mach in names and st["rc"] == 0 and fr["frames"] == [0, 2, 4, 6] and diff == 0 and extra_ok
        print(f"  {label}: exit {st['rc']}, frames {fr['frames']}, "
              f"max difference to the local run {diff:.1e}  {'OK' if ok else 'FAIL'}")
finally:
    srv.terminate()
