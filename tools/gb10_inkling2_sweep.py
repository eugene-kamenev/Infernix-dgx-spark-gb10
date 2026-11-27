#!/usr/bin/env python3
"""GB10 (sm_121a) Kernel & Schedule Sweep Driver for Inkling 2 Projections.

Modes:
  - baseline: builds and runs `infernix_inkling2_sweep` and prints a GB10 roofline summary.
  - sweep:    tests candidate `dmma_splits` target-warp caps, DMMA ahead depths, and
              `projections_mma` wide prefill schedules inside `src/ops/inkling2/projections_mma.cu`,
              enforces STACK=0 and LOCAL=0 via `cuobjdump -res-usage`, restores the source cleanly,
              and prints the ranked winners.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build-native"
BENCH_BIN = BUILD_DIR / "bench" / "infernix_inkling2_sweep"
PROJECTIONS_MMA = ROOT / "src" / "ops" / "inkling2" / "projections_mma.cu"


def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=check)


def build_sweep_bin() -> None:
    res = run_cmd(["ninja", "-C", str(BUILD_DIR), "-j", "20", "infernix_inkling2_sweep"], check=False)
    if res.returncode != 0:
        raise RuntimeError(f"build failed:\n{res.stdout}\n{res.stderr}")


def check_inkling2_spills() -> List[str]:
    res = run_cmd(["cuobjdump", "-res-usage", str(BENCH_BIN)])
    bad: List[str] = []
    cur_func = None
    for line in res.stdout.splitlines():
        if line.strip().startswith("Function "):
            cur_func = line.strip().split("Function ", 1)[1].rstrip(":")
        elif "STACK:" in line or "LOCAL:" in line:
            if cur_func and ("projections_dmma_kernel" in cur_func or "projections_mma_w4_kernel" in cur_func):
                m_st = re.search(r"STACK:(\d+)", line)
                m_lo = re.search(r"LOCAL:(\d+)", line)
                st = int(m_st.group(1)) if m_st else 0
                lo = int(m_lo.group(1)) if m_lo else 0
                if st > 0 or lo > 0:
                    bad.append(f"{cur_func}: STACK={st}, LOCAL={lo}")
    return bad


def run_bench(shapes: str = "all", fmt: str = "all", tokens: str = "1,2,4,8,16,128,1024") -> List[Dict[str, str]]:
    res = run_cmd([str(BENCH_BIN), "--csv", "--filter", shapes, "--fmt", fmt, "--tokens", tokens])
    reader = csv.DictReader(io.StringIO(res.stdout))
    return list(reader)


def mode_baseline() -> None:
    build_sweep_bin()
    rows = run_bench()
    print(f"{'shape':<12} {'fmt':<4} {'route':<7} {'T':>5} {'us':>9} {'GB/s':>8} {'%BW(244.4)':>10} {'TFLOP/s':>8}")
    print("-" * 70)
    for r in rows:
        gbps = float(r["gbps"])
        pct_bw = 100.0 * gbps / 244.4
        print(
            f"{r['shape']:<12} {r['fmt']:<4} {r['route']:<7} {int(r['T']):>5} "
            f"{float(r['us']):>9.2f} {gbps:>8.2f} {pct_bw:>9.1f}% {float(r['tflops']):>8.3f}"
        )


def replace_once(src: str, target: str, repl: str) -> str:
    if src.count(target) != 1:
        raise RuntimeError(f"Expected 1 occurrence of target, found {src.count(target)}: {target[:60]!r}")
    return src.replace(target, repl, 1)


def mode_sweep() -> None:
    orig = PROJECTIONS_MMA.read_text()
    try:
        build_sweep_bin()
        print("=== Baseline spills check (DMMA + W4 MMA) ===")
        spills = check_inkling2_spills()
        print("OK (0 spills)" if not spills else f"SPILLS: {spills}")

        # 1. Sweep dmma_splits target warps (baseline = 2048 for RTX 5090 170 SMs; GB10 has 48 SMs)
        print("\n=== Sweep 1: dmma_splits target warps (T in {1, 2, 4, 8, 16}, shapes=k2,w_o, fmt=all) ===")
        dmma_target = "while (splits < 16 && tiles * splits < 2048 && blocks % (2 * splits) == 0 && blocks / (2 * splits) >= 4) {"
        dmma_results: Dict[int, List[Dict[str, str]]] = {}
        for warps in [2048, 1024, 768, 512, 384]:
            PROJECTIONS_MMA.write_text(
                orig if warps == 2048 else replace_once(orig, dmma_target, dmma_target.replace("2048", str(warps)))
            )
            build_sweep_bin()
            rows_k2 = run_bench(shapes="k2", fmt="all", tokens="1,2,4,8,16")
            rows_wo = run_bench(shapes="w_o", fmt="all", tokens="1,2,4,8,16")
            dmma_results[warps] = rows_k2 + rows_wo
            tot_us = sum(float(r["us"]) for r in dmma_results[warps])
            w4_us = sum(float(r["us"]) for r in dmma_results[warps] if r["fmt"] == "w4")
            print(f"target_warps={warps:>4}: total_us={tot_us:8.2f}, w4_us={w4_us:8.2f}")

        # Print per-(shape,fmt,T) winners vs 2048
        base_map = {(r["shape"], r["fmt"], r["T"]): float(r["us"]) for r in dmma_results[2048]}
        print("\nPer-point DMMA winners vs 2048 baseline:")
        for key, b_us in base_map.items():
            best_w, best_us = min(
                ((w, float(r["us"])) for w, rs in dmma_results.items() for r in rs if (r["shape"], r["fmt"], r["T"]) == key),
                key=lambda x: x[1],
            )
            pct = (b_us - best_us) / b_us * 100.0
            if best_w != 2048 and pct >= 1.0:
                print(f"  {key[0]:<5} {key[1]:<4} T={key[2]:>2}: best_warps={best_w:>4} ({best_us:6.2f} us vs {b_us:6.2f} us, {pct:+.1f}%)")

        # 2. Sweep kDmmaW4Ahead (baseline = 12)
        print("\n=== Sweep 2: kDmmaW4Ahead (T in {1, 8}, shapes=k2,w_o, fmt=w4) ===")
        ahead_target = "constexpr int kDmmaW4Ahead   = 12;"
        for ahead in [12, 8, 10, 16]:
            PROJECTIONS_MMA.write_text(
                orig if ahead == 12 else replace_once(orig, ahead_target, f"constexpr int kDmmaW4Ahead   = {ahead};")
            )
            build_sweep_bin()
            spills = check_inkling2_spills()
            rows = run_bench(shapes="k2", fmt="w4", tokens="1,8") + run_bench(shapes="w_o", fmt="w4", tokens="1,8")
            s_us = sum(float(r["us"]) for r in rows)
            detail = ", ".join(f"{r['shape']}@{r['T']}={float(r['us']):.1f}us" for r in rows)
            print(f"kDmmaW4Ahead={ahead:>2}: sum={s_us:6.2f} us [{detail}] spills={len(spills)}")

        # 3. Sweep wide prefill schedule (projections_mma) at T in {128, 1024}
        print("\n=== Sweep 3: MMA prefill wide Schedule (T in {128, 1024}, shapes=k2,w_o,dspark_gu,dspark_down) ===")
        sched_target = "        using Sched = Schedule<128, 128, 32, 32, 3, 1>;"
        sched_candidates = [
            ("128x128_3s_1b", "        using Sched = Schedule<128, 128, 32, 32, 3, 1>;"),
            ("128x128_2s_1b", "        using Sched = Schedule<128, 128, 32, 32, 2, 1>;"),
            ("64x128_2s_2b", "        using Sched = Schedule<64, 128, 32, 32, 2, 2>;"),
            ("128x64_2s_2b", "        using Sched = Schedule<128, 64, 32, 32, 2, 2>;"),
            ("64x64_3s_2b", "        using Sched = Schedule<64, 64, 32, 32, 3, 2>;"),
            ("64x64_2s_2b", "        using Sched = Schedule<64, 64, 32, 32, 2, 2>;"),
        ]
        mma_results: Dict[str, List[Dict[str, str]]] = {}
        for label, line in sched_candidates:
            PROJECTIONS_MMA.write_text(orig if label == "128x128_3s_1b" else replace_once(orig, sched_target, line))
            build_sweep_bin()
            spills = check_inkling2_spills()
            rows: List[Dict[str, str]] = []
            for sh in ["k2", "w_o", "dspark_gu", "dspark_down"]:
                rows.extend(run_bench(shapes=sh, fmt="all", tokens="128,1024"))
            mma_results[label] = rows
            t128 = sum(float(r["us"]) for r in rows if r["T"] == "128")
            t1024 = sum(float(r["us"]) for r in rows if r["T"] == "1024")
            print(f"{label:<15}: T=128 sum={t128:8.1f} us, T=1024 sum={t1024:8.1f} us, spills={len(spills)}")

        base_mma = {(r["shape"], r["fmt"], r["T"]): float(r["us"]) for r in mma_results["128x128_3s_1b"]}
        for T_str in ["128", "1024"]:
            print(f"\nBest MMA schedules at T={T_str} vs 128x128_3s_1b:")
            for sh in ["k2", "w_o", "dspark_gu", "dspark_down"]:
                for fm in ["bf16", "q8", "w4"]:
                    key = (sh, fm, T_str)
                    b_us = base_mma[key]
                    best_lbl, best_us = min(
                        ((lbl, float(r["us"])) for lbl, rs in mma_results.items() for r in rs if (r["shape"], r["fmt"], r["T"]) == key),
                        key=lambda x: x[1],
                    )
                    pct = (b_us - best_us) / b_us * 100.0
                    print(f"  {sh:<11} {fm:<4}: {best_lbl:<15} {best_us:7.1f} us vs {b_us:7.1f} us ({pct:+.1f}%)")
    finally:
        PROJECTIONS_MMA.write_text(orig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "sweep"], default="baseline")
    args = ap.parse_args()
    if args.mode == "baseline":
        mode_baseline()
    else:
        mode_sweep()
    return 0


if __name__ == "__main__":
    sys.exit(main())
