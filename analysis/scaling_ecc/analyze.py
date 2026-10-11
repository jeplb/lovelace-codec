"""recompute lovelace physical accounting and computational correction."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.scaling_ecc.scaling import build
from analysis.scaling_ecc.sequence import accelerated, run_simulation
from analysis.scaling_ecc.unequal import run as unequal
from analysis.scaling_ecc.retrospective import analyze as retrospective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--ecc", action="store_true")
    parser.add_argument("--full-simulation", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    all_steps = not any((args.scaling, args.ecc, args.benchmark))
    if args.scaling or all_steps:
        build.main()
    if args.ecc or all_steps:
        accelerated.build()
        run_simulation.run("lovelace")
        run_simulation.summarize()
        unequal.main()
        retrospective.main()
    if args.full_simulation:
        for name in ("realizations_lovelace.csv", "predictions_lovelace.npz", "decoder_diagnostics_lovelace.tsv"):
            path = Path(run_simulation.HERE) / name
            if path.exists():
                path.unlink()
        run_simulation.run("lovelace")
        run_simulation.summarize()
    if args.benchmark:
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "benchmark/run.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
