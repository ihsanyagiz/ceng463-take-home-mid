"""Run the full study end-to-end in the required order.

    python run_all.py              # run everything (final test refuses if already done)
    python run_all.py --clean      # delete results/ and generated report tables/figures first
    python run_all.py --force-final  # allow re-running the single test evaluation (logged)
    python run_all.py --skip-final   # everything except the test evaluation
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["src.environment", "src.q1_data_leakage", "src.q2_classifiers", "src.q3_regression",
         "src.q4_trees_ensembles", "src.q5_robustness_selection", "src.q5_final_test", "src.export_report_assets"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--force-final", action="store_true")
    ap.add_argument("--skip-final", action="store_true",
                    help="run everything except the single test evaluation (src.q5_final_test)")
    args = ap.parse_args()
    if args.clean:
        for d in (ROOT / "results", ROOT / "report" / "tables", ROOT / "report" / "figures"):
            shutil.rmtree(d, ignore_errors=True)
    timings = {}
    t0 = time.perf_counter()
    for step in STEPS:
        if step == "src.q5_final_test" and args.skip_final:
            print("\n===== skipping src.q5_final_test (--skip-final) =====")
            continue
        cmd = [sys.executable, "-W", "ignore", "-m", step]
        if step == "src.q5_final_test" and args.force_final:
            cmd.append("--force")
        print(f"\n===== {step} =====", flush=True)
        ts = time.perf_counter()
        r = subprocess.run(cmd, cwd=ROOT)
        timings[step] = round(time.perf_counter() - ts, 1)
        if r.returncode != 0:
            print(f"Step {step} failed (exit {r.returncode}); stopping.")
            sys.exit(r.returncode)
    timings["TOTAL"] = round(time.perf_counter() - t0, 1)
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "logs" / "runtime.json").write_text(json.dumps(timings, indent=2))
    print("\nRuntime (s):", json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
