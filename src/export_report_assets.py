"""Copy generated PDF figures into report/figures (tables are already written to report/tables)."""
import shutil

from . import config as C


def main():
    n = 0
    for f in C.FIGURES.glob("*.pdf"):
        shutil.copy2(f, C.REPORT_FIGURES / f.name)
        n += 1
    print(f"copied {n} figures to {C.REPORT_FIGURES}")


if __name__ == "__main__":
    main()
