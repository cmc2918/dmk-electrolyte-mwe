#!/usr/bin/env python3
"""
Two-week chain of TorchSim NVT segments for the 600-atom 1.5 M LiFSI/DMK box (combined FAN+DMK model), each run by
run_torchsim_dmk600.py with its bond watchdog. When the watchdog stops a segment (an S-F bond broken, exit code 2),
the chain goes back REWIND_PS before the event, draws new Maxwell-Boltzmann velocities (new seed) and continues from
there, until the wall-clock budget is used up. One trajectory: every segment continues the same configurational path.

Start: 100 ps before the second S-F break, i.e. frame 2891.5 ps of torchsim_dmk600_combined_restart.h5, which is
8421.0 ps along the path of solvation_stitched_1.0-8.5ns (run 1 to 5529.5 ps, then the 2026-09-25 restart).

Rewind rule: REWIND_PS (100 ps) before the break; REWIND_LONG_PS (200 ps) if the segment broke before getting past the
point where its parent broke (within its own first rewind + 50 ps). A rewind longer than the segment walks back into
its parent. After MAX_QUICK consecutive such quick breaks the chain stops (something is wrong, not just bad luck).
A segment that completes (budget used, or ends early because the GPU was faster than assumed) is continued from its
last frame if more than an hour is left; an interrupted chain (Ctrl-C, reboot) is continued with --resume.

Files: chain_sNN.{h5,dcd,pdb,out}, chain.log, chain_manifest.json. The manifest lists every segment with its parent,
the parent time it branches from (its frame 0 = that parent frame), seed, status, end time and path offset; the
analysis keeps, for each segment, frames 1 .. the branch point of its child (or its end), as for the first restart.

    tmux new -s mlff_chain
    bash run_chain.sh --hours 336                 # in DMK_600_torchsim_test; progress: tail -f chain.log
    bash run_chain.sh --resume --hours <left>     # after an interruption
    bash run_chain.sh --smoke                     # CPU code-path test: 20-step segments with forced aborts
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = str(HERE / "LiFSI_FAN_DMK_mh0.model")                         # the deposited MACE-FT model
DRIVER = HERE / "run_torchsim_dmk600.py"
TIMESTEP_PS = 0.001
REWIND_PS, REWIND_LONG_PS = 100.0, 200.0
MAX_QUICK = 6
SEED0 = 3000
MS_PER_STEP = 41.6          # both earlier production segments: 65.00 h / 5.6295 M steps, 34.56 h / 2.991 M steps
MARGIN_S = 300              # warm-up (~90 s) and file writing per segment
MIN_SEGMENT_NS = 0.05

# the two earlier segments, as roots of the path (times in each file's own step count)
ROOTS = [
    {"tag": "torchsim_dmk600_combined_10ns", "parent": None, "branch_ps": None, "offset_ps": 0.0, "status": "root"},
    {"tag": "torchsim_dmk600_combined_restart", "parent": "torchsim_dmk600_combined_10ns", "branch_ps": 5529.5,
     "offset_ps": 5529.5, "status": "root", "end_ps": 2991.5},
]
FIRST_START = ("torchsim_dmk600_combined_restart", 2891.5, REWIND_PS)   # parent, branch time, rewind it represents


def last_frame_ps(h5):
    with h5py.File(h5, "r") as f:
        return float(np.ravel(f["steps/positions"][:])[-1]) * TIMESTEP_PS


class Chain:
    def __init__(self, a):
        self.a, self.smoke = a, a.smoke
        self.dir = HERE / "smoke_test" if self.smoke else HERE
        self.dir.mkdir(exist_ok=True)
        self.manifest = self.dir / "chain_manifest.json"
        self.logfile = self.dir / "chain.log"
        self.rewind, self.rewind_long = (0.01, 0.02) if self.smoke else (REWIND_PS, REWIND_LONG_PS)
        self.min_keep = 0.005 if self.smoke else 0.5      # a branch point must leave at least frame 1 of the segment
        self.segs = json.load(open(self.manifest))["segments"] if a.resume else [dict(s) for s in ROOTS]
        self.deadline = time.time() + a.hours * 3600.0
        self.ms_per_step = MS_PER_STEP

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.logfile, "a") as f:
            f.write(line + "\n")

    def save(self):
        json.dump({"note": __doc__.split("\n\n")[0], "timestep_ps": TIMESTEP_PS, "segments": self.segs},
                  open(self.manifest, "w"), indent=1)

    def seg(self, tag):
        return next(s for s in self.segs if s["tag"] == tag)

    def h5(self, tag):
        root = any(r["tag"] == tag for r in ROOTS)
        return (HERE if root else self.dir) / f"{tag}.h5"

    def rewind_point(self, tag, t, back):
        """Walk back `back` ps from time t of segment `tag`, into its parents if needed."""
        while t - back < self.min_keep:
            s = self.seg(tag)
            if s["parent"] is None:
                raise RuntimeError("rewind reached the start of run 1")
            back -= t
            tag, t = s["parent"], s["branch_ps"]
        return tag, t - back

    def run_segment(self, parent, branch_ps, rewind_used):
        k = sum(1 for s in self.segs if s["tag"].startswith("chain_s")) + 1
        tag, seed = f"chain_s{k:02d}", SEED0 + k
        left = self.deadline - time.time() - MARGIN_S
        ns = 0.0 if self.smoke else left / (self.ms_per_step * 1e-3) * TIMESTEP_PS * 1e-3
        if not self.smoke and ns < MIN_SEGMENT_NS:
            self.log(f"budget used up ({left / 3600:.2f} h left): chain finished")
            return None
        p = self.seg(parent)
        with h5py.File(self.h5(parent), "r") as f:          # the driver starts from the nearest stored frame
            steps = np.ravel(f["steps/positions"][:])
        branch_ps = float(steps[np.argmin(np.abs(steps - branch_ps / TIMESTEP_PS))]) * TIMESTEP_PS
        s = {"tag": tag, "parent": parent, "branch_ps": branch_ps, "offset_ps": p["offset_ps"] + branch_ps,
             "seed": seed, "rewind_used_ps": rewind_used, "requested_ns": round(ns, 4), "status": "running",
             "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.segs.append(s)
        self.save()
        cmd = [sys.executable, str(DRIVER), "--model", MODEL, "--start-h5", str(self.h5(parent)),
               "--start-ps", f"{branch_ps:.3f}", "--seed", str(seed), "--equil-ps", "10", "--tag", tag]
        cmd += ["--smoke", "--device", "cpu"] if self.smoke else ["--ns", f"{ns:.4f}"]
        self.log(f"{tag}: from {parent} at {branch_ps:.1f} ps (path {s['offset_ps']:.1f} ps), seed {seed}, "
                 + ("smoke" if self.smoke else f"{ns:.3f} ns requested ({left / 3600:.1f} h left)"))
        t0 = time.time()
        with open(self.dir / f"{tag}.out", "w") as out:
            rc = subprocess.call(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
        wall = time.time() - t0
        text = (self.dir / f"{tag}.out").read_text(errors="replace")
        end = last_frame_ps(self.h5(tag)) if self.h5(tag).exists() else None
        s.update({"exit_code": rc, "end_ps": end, "wall_h": round(wall / 3600, 3),
                  "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        m = re.search(r"ABORTED by watchdog: (.*?) ===", text)
        s["status"] = {0: "completed", 2: "broken"}.get(rc, "error")
        if m:
            s["break"] = m.group(1)
        if end and end > 500 and not self.smoke:            # measured speed for the next budget
            self.ms_per_step = wall / (end / TIMESTEP_PS) * 1e3
            s["ms_per_step"] = round(self.ms_per_step, 2)
        self.save()
        self.log(f"{tag}: {s['status']} at {end} ps after {wall / 3600:.2f} h" + (f" -- {s['break']}" if m else ""))
        return s

    def next_start(self, s, quick_count):
        """(parent, branch time, rewind used, quick-break count) for the segment after s."""
        if s["status"] == "broken":
            quick = s["end_ps"] < s["rewind_used_ps"] + (0.005 if self.smoke else 50.0)
            back = self.rewind_long if quick else self.rewind
            parent, t = self.rewind_point(s["tag"], s["end_ps"], back)
            self.log(f"  rewind {back:g} ps -> {parent} at {t:.1f} ps" + ("  (quick break: long rewind)" if quick else ""))
            return parent, t, back, quick_count + 1 if quick else 0
        return s["tag"], s["end_ps"], 0.0, 0              # completed or interrupted: continue from the last frame

    def run(self):
        self.log(f"chain {'resumed' if self.a.resume else 'started'}: budget {self.a.hours:.1f} h, "
                 f"rewind {self.rewind:g}/{self.rewind_long:g} ps, {self.ms_per_step} ms/step assumed")
        if self.a.resume:
            last = self.segs[-1]
            if last["status"] == "running":                 # killed mid-segment: keep what it wrote
                last.update({"status": "interrupted", "end_ps": last_frame_ps(self.h5(last["tag"]))})
                self.save()
            start = self.next_start(last, 0)
        else:
            start = (*FIRST_START, 0)
        n = 0
        while True:
            parent, branch, back, quick = start
            if quick >= MAX_QUICK:
                self.log(f"stopping: {quick} consecutive quick breaks")
                break
            s = self.run_segment(parent, branch, back)
            if s is None:
                break
            n += 1
            if s["status"] == "error" or s["end_ps"] is None:
                self.log(f"stopping: {s['tag']} failed (exit {s['exit_code']}); see {s['tag']}.out")
                break
            if self.smoke and n >= self.a.max_segments:
                break
            if s["status"] == "completed" and not self.smoke and self.deadline - time.time() < 3600:
                self.log("budget used up: chain finished")
                break
            start = self.next_start(s, quick)
        self.log(f"done: {n} segments this session; manifest {self.manifest.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=336.0, help="wall-clock budget (default 14 days)")
    ap.add_argument("--resume", action="store_true", help="continue the chain recorded in chain_manifest.json")
    ap.add_argument("--smoke", action="store_true", help="CPU code-path test in smoke_test/ (forced aborts)")
    ap.add_argument("--max-segments", type=int, default=4, help="segments in --smoke mode")
    a = ap.parse_args()
    Chain(a).run()


if __name__ == "__main__":
    main()
