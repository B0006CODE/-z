from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent.parent

SCRIPTS = [
    "extract_tables.py",
    "fig2_main_results.py",
    "fig3_recall_at_k.py",
    "fig4_structural_ablation.py",
    "fig5_subgroup_forest.py",
    "fig6_efficiency_pareto.py",
    "fig7_interpretability.py",
    "fig8_synonym_normalization.py",
]

EXPECTED = [
    "fig2_main_results",
    "fig3_recall_at_k",
    "fig4_structural_ablation",
    "fig5_subgroup_forest",
    "fig6_efficiency_pareto",
    "fig7_interpretability",
    "fig8_synonym_normalization",
]


def run_script(script: str) -> None:
    path = SCRIPT_DIR / script
    print(f"[make_all] running {path}")
    subprocess.run([sys.executable, str(path)], cwd=REPO_ROOT, check=True)


def verify_outputs() -> None:
    missing = []
    for name in EXPECTED:
        for ext in ("pdf", "svg", "png"):
            path = PROJECT_ROOT / "figures" / ext / f"{name}.{ext}"
            if not path.exists() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing expected figure outputs:\n" + "\n".join(missing))
    print("[make_all] verified all expected PDF, SVG, and PNG outputs")


def main() -> None:
    for script in SCRIPTS:
        run_script(script)
    verify_outputs()


if __name__ == "__main__":
    main()

