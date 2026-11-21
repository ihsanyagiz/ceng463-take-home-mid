"""Write results/environment.txt (hardware/software summary) for the report."""
import platform
import sys
from importlib.metadata import version

import psutil

from . import config as C

PKGS = ["numpy", "scipy", "pandas", "scikit-learn", "imbalanced-learn", "matplotlib", "seaborn", "ucimlrepo",
        "joblib"]


def main():
    lines = [
        f"OS: {platform.platform()}",
        f"CPU: {platform.processor()} | physical cores: {psutil.cpu_count(logical=False)} | "
        f"logical cores: {psutil.cpu_count(logical=True)}",
        f"RAM: {psutil.virtual_memory().total / 2**30:.1f} GiB",
        f"Python: {sys.version.split()[0]}",
        *[f"{p}: {version(p)}" for p in PKGS],
        f"SEED: {C.SEED} | CV folds: {C.CV_FOLDS} | n_jobs: {C.N_JOBS}",
    ]
    (C.RESULTS / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = "\\begin{verbatim}\n" + "\n".join(lines) + "\n\\end{verbatim}\n"
    (C.REPORT_TABLES / "environment.tex").write_text(tex, encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
