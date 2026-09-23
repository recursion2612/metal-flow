"""Qiskit Metal and SQDMetal adapters for the optimization pipeline."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from .palace import (
    E_CHARGE,
    H_PLANCK,
    PHI_0,
    max_mpi_procs,
    update_qiskit_geometry,
)


def create_transmon_design(
    *,
    chip_size_x: str = "2.8mm",
    chip_size_y: str = "2mm",
    q1_name: str = "Q1",
    q1_options: dict[str, Any] | None = None,
):
    """Create the Qiskit Metal transmon used by the default optimization bounds."""
    try:
        from qiskit_metal import designs
        from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
    except ImportError as exc:
        raise ImportError(
            "Qiskit Metal is required. Install the local quantum-metal package."
        ) from exc

    design = designs.DesignPlanar({}, overwrite_enabled=True)
    design.chips.main.size["size_x"] = chip_size_x
    design.chips.main.size["size_y"] = chip_size_y
    options = {
        "pad_width": "455um",
        "pad_height": "90um",
        "pad_gap": "30um",
        "pocket_width": "1.2mm",
        "pocket_height": "650um",
        "connection_pads": {},
    }
    if q1_options:
        options.update(q1_options)
    TransmonPocket(design, q1_name, options=options)
    design.rebuild()
    return design


def export_qiskit_metal_gmsh(design, output_path: str) -> str:
    """Export a populated Gmsh mesh using the local Qiskit Metal renderer."""
    try:
        from qiskit_metal.renderers.renderer_gmsh.gmsh_renderer import QGmshRenderer
    except ImportError as exc:
        raise ImportError(
            "Qiskit Metal Gmsh support is required. Install quantum-metal[mesh]."
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = QGmshRenderer(design)
    try:
        try:
            renderer.render_design(mesh_geoms=False)
            renderer.add_mesh(dim=3)
            renderer.export_mesh(str(output))
        except AttributeError as exc:
            raise RuntimeError(
                "The local QGmshRenderer is incompatible with this DesignPlanar "
                "layer stack; use SqdmetalCapacitanceRunner for Palace meshes."
            ) from exc
    finally:
        renderer.close()
    return str(output)


class SqdmetalCapacitanceRunner:
    """Generate and run a Palace capacitance simulation through SQDMetal."""

    def __init__(
        self,
        output_root: str,
        palace_bin: str,
        n_procs: int = 4,
        dielectric_material: str = "silicon",
        solver_order: int = 2,
        retain_visualization: bool = False,
    ) -> None:
        if not isinstance(n_procs, int) or not 1 <= n_procs <= max_mpi_procs():
            raise ValueError(
                f"Palace MPI processes must be between 1 and {max_mpi_procs()} "
                "(90% of logical CPUs, rounded up)"
            )
        self.output_root = Path(output_root)
        self.palace_bin = palace_bin
        self.n_procs = n_procs
        self.dielectric_material = dielectric_material
        self.solver_order = solver_order
        self.retain_visualization = retain_visualization

    def evaluate(self, design, params: np.ndarray, run_name: str) -> tuple[float, float]:
        """Run Palace and return measured ``(Ej_MHz, Ec_MHz)``."""
        try:
            from SQDMetal.PALACE.Capacitance_Simulation import (
                PALACE_Capacitance_Simulation,
            )
        except ImportError as exc:
            raise ImportError(
                "SQDMetal is required for the production Palace backend."
            ) from exc

        run_root = self.output_root / run_name
        run_root.parent.mkdir(parents=True, exist_ok=True)
        options = {
            "palace_dir": self.palace_bin,
            "num_cpus": self.n_procs,
            "solver_order": self.solver_order,
            "dielectric_material": self.dielectric_material,
            "palace_mode": "local",
            "solver_tol": 1.0e-8,
            "solver_maxits": 250,
        }
        simulation = PALACE_Capacitance_Simulation(
            name=run_root.name,
            sim_parent_directory=str(run_root.parent) + "/",
            mode="PC",
            meshing="GMSH",
            user_options=options,
            metal_design=design,
            create_files=True,
        )
        simulation.add_metallic(1)
        simulation.add_ground_plane()
        simulation.fine_mesh_features(
            100e-6,
            min_size=12e-6,
            max_size=100e-6,
            taper_dist_min=10e-6,
            taper_dist_max=200e-6,
        )
        try:
            import gmsh
            if not gmsh.isInitialized():
                gmsh.initialize()
        except ImportError as exc:
            raise ImportError("gmsh is required for SQDMetal Palace meshing") from exc
        simulation.prepare_simulation()
        if gmsh.isInitialized():
            gmsh.finalize()
        simulation.run()

        result_file = Path(simulation._output_data_dir) / "terminal-C.csv"
        if not result_file.exists():
            raise FileNotFoundError(f"SQDMetal did not produce {result_file}")
        matrix = _read_terminal_capacitance(result_file)
        if not self.retain_visualization:
            shutil.rmtree(result_file.parent / "paraview", ignore_errors=True)
            for image_path in result_file.parent.glob("*.png"):
                image_path.unlink(missing_ok=True)
        c_sigma = float(matrix[0, 0])
        if not np.isfinite(c_sigma) or c_sigma <= 0:
            raise ValueError(f"Invalid qubit self-capacitance in {result_file}: {c_sigma}")
        lj_val = float(params[3])
        if not np.isfinite(lj_val) or lj_val <= 0:
            raise ValueError("Josephson inductance must be positive and finite")
        ec_mhz = (E_CHARGE**2 / (2.0 * c_sigma * H_PLANCK)) * 1e-6
        ej_mhz = (PHI_0**2 / (4.0 * np.pi**2 * lj_val * H_PLANCK)) * 1e-6
        return float(ej_mhz), float(ec_mhz)


def _read_terminal_capacitance(path: Path) -> np.ndarray:
    """Read SQDMetal's terminal-C.csv, whose first column is the row index."""
    frame = pd.read_csv(path, index_col=0)
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"No finite capacitance matrix found in {path}")
    if values.shape[0] != values.shape[1]:
        raise ValueError(f"Capacitance matrix in {path} is not square: {values.shape}")
    return values
