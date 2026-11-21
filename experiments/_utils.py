"""Small helpers shared by the experiment scripts (path setup, markdown tables, plotting)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401

OUTPUTS = ROOT / "outputs"


def md_table(rows: list[dict], cols: list[str] | None = None, floatfmt: str = ".4f") -> str:
    """Render a list of dicts as a GitHub markdown table (no extra dependencies)."""
    if not rows:
        return "_(no rows)_"
    cols = cols or list(rows[0])
    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)
