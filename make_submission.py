"""Build CENG463_Midterm_<StudentID>.zip (code, report source + PDF, results, README, requirements).

    python make_submission.py --student-id 123456789
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INCLUDE = ["README.md", "requirements.txt", "LLM_USAGE_LOG.md", "run_all.py", "make_submission.py",
           "CENG463_TakeHome.md", "src", "notebooks", "report", "results", "data/splits"]
EXCLUDE_PARTS = {".venv", "__pycache__", ".ipynb_checkpoints", ".git"}
EXCLUDE_SUFFIX = {".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz", ".bbl", ".blg", ".pyc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True)
    args = ap.parse_args()
    out = ROOT / f"CENG463_Midterm_{args.student_id}.zip"
    if not (ROOT / "report" / "main.pdf").exists():
        print("WARNING: report/main.pdf not found - compile the report first.")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else (sorted(p.rglob("*")) if p.exists() else [])
            for f in files:
                rel = f.relative_to(ROOT)
                if f.is_dir() or EXCLUDE_PARTS & set(f.parts) or f.suffix in EXCLUDE_SUFFIX - {".log"}:
                    continue
                # .log is a LaTeX build artefact only under report/; results/logs/*.log are kept
                if f.suffix == ".log" and rel.parts[0] == "report":
                    continue
                z.write(f, f.relative_to(ROOT))
                n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 2**20:.1f} MiB). Raw data and .venv excluded.")


if __name__ == "__main__":
    main()
