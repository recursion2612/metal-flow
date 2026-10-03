"""Visualize and export Qiskit Metal transmon designs to PNG."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cad import create_transmon_design, update_qiskit_geometry


def render_transmon(
    pad_width: float,
    pad_height: float,
    pad_gap: float,
    lj: float | None = None,
    output_png: str = "transmon_design.png",
    title: str | None = None,
    dpi: int = 300,
) -> Path:
    """Render a transmon design to PNG."""
    from qiskit_metal import view

    design = create_transmon_design()
    params = [float(pad_width), float(pad_height), float(pad_gap), float(lj or 10.0e-9)]
    param_keys = ["Q1.pad_width", "Q1.pad_height", "Q1.pad_gap", "lj"]
    update_qiskit_geometry(design, params, param_keys)

    fig = view(design)
    ax = fig.gca()
    if title:
        ax.set_title(title, fontsize=12, pad=12)

    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a transmon design from optimization_result.json or parameters to PNG."
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default=None,
        help="Path to optimization_result.json",
    )
    parser.add_argument("--pad-width", type=float, default=None, help="Pad width in um")
    parser.add_argument("--pad-height", type=float, default=None, help="Pad height in um")
    parser.add_argument("--pad-gap", type=float, default=None, help="Pad gap in um")
    parser.add_argument("--lj", type=float, default=10.0e-9, help="Josephson inductance in H")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="final_design.png",
        help="Output PNG path (default: final_design.png)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI (default: 300)")

    args = parser.parse_args()

    title = None
    if args.result_json:
        json_path = Path(args.result_json)
        if not json_path.exists():
            sys.exit(f"Error: {json_path} does not exist")
        data = json.loads(json_path.read_text())
        best_p = data.get("best_parameters", {})
        pad_width = best_p.get("Q1.pad_width", args.pad_width)
        pad_height = best_p.get("Q1.pad_height", args.pad_height)
        pad_gap = best_p.get("Q1.pad_gap", args.pad_gap)
        lj = best_p.get("lj", args.lj)

        metrics = data.get("best_metrics_mhz", {})
        ej = metrics.get("Ej", None)
        ec = metrics.get("Ec", None)
        if ej and ec:
            title = f"Transmon Design (Ej={ej:.1f} MHz, Ec={ec:.1f} MHz, Lj={lj*1e9:.2f} nH)"
    else:
        if args.pad_width is None or args.pad_height is None or args.pad_gap is None:
            sys.exit("Error: Must specify either --result-json or (--pad-width, --pad-height, --pad-gap)")
        pad_width = args.pad_width
        pad_height = args.pad_height
        pad_gap = args.pad_gap
        lj = args.lj
        title = f"Transmon Design (w={pad_width:.1f}um, h={pad_height:.1f}um, gap={pad_gap:.1f}um)"

    out_file = render_transmon(
        pad_width=pad_width,
        pad_height=pad_height,
        pad_gap=pad_gap,
        lj=lj,
        output_png=args.output,
        title=title,
        dpi=args.dpi,
    )
    print(f"Design saved to: {out_file}")


if __name__ == "__main__":
    main()

