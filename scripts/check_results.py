"""Compare regenerated headlines to committed results with floating tolerance."""
import json
import math
from pathlib import Path
import sys


def compare(a, b, path="root"):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            compare(a[key], b[key], path + "." + key)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + f"[{i}]")
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10), (path, a, b)
    else:
        assert a == b, (path, a, b)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    compare(json.loads((root / "results/summary.json").read_text()),
            json.loads(Path(sys.argv[1]).read_text()))
    print("Regenerated headline results match the committed snapshot.")
