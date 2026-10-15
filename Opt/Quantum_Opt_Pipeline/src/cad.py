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
    default_mpi_procs,
    max_mpi_procs,
    update_qiskit_geometry,
)


def create_transmon_design(
    *,
    chip_size_x: str = "2.8mm",
    chip_size_y: str = "2mm",
    q1_name: str = "Q1",
    q1_options: dict[str, Any] | None = None,
    include_readout_pad: bool = False,
    connection_pads: dict[str, Any] | None = None,
):
    """Create the Qiskit Metal transmon used by the optimization bounds.
    
    Supports optional readout connection pads to enable direct multi-terminal
    capacitance matrix extraction, coupling rate (g), and dispersive shift (chi).
    """
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

    conn_pads = {}
    if include_readout_pad:
        conn_pads["readout"] = {
            "connector_type": "0",
            "pad_width": "125um",
            "pad_height": "30um",
            "pad_gap": "15um",
            "pad_cpw_shift": "0um",
            "pad_cpw_extent": "25um",
            "cpw_width": "10um",
            "cpw_gap": "6um",
            "cpw_extend": "100um",
            "loc_W": 1,
            "loc_H": 1,
        }
    if connection_pads:
        conn_pads.update(connection_pads)
    options["connection_pads"] = conn_pads

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
        n_procs: int | None = None,
        dielectric_material: str = "silicon",
        solver_order: int = 2,
        retain_visualization: bool = False,
    ) -> None:
        if n_procs is None:
            n_procs = default_mpi_procs()
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
        c_sigma = extract_transmon_c_sigma(matrix)
        if not np.isfinite(c_sigma) or c_sigma <= 0:
            raise ValueError(f"Invalid qubit self-capacitance in {result_file}: {c_sigma}")
        lj_val = float(params[3])
        if not np.isfinite(lj_val) or lj_val <= 0:
            raise ValueError("Josephson inductance must be positive and finite")
        ec_mhz = (E_CHARGE**2 / (2.0 * c_sigma * H_PLANCK)) * 1e-6
        ej_mhz = (PHI_0**2 / (4.0 * np.pi**2 * lj_val * H_PLANCK)) * 1e-6
        return float(ej_mhz), float(ec_mhz)


def extract_transmon_c_sigma(matrix: np.ndarray) -> float:
    """
    Extract the effective charging capacitance C_sigma for a floating transmon
    from the Maxwell capacitance matrix (from AWS Palace / SQDMetal).

    Supports:
      - 3-terminal floating transmon without connection pads:
          Index 0: Ground plane ("Chip")
          Index 1: Transmon Pad 1 ("Cond1")
          Index 2: Transmon Pad 2 ("Cond2")
          C_sigma = C_12 + (C_1_g * C_2_g) / (C_1_g + C_2_g)
      - (N+3)-terminal transmon with N connection pads (readout, bus):
          Indices 3..N+2: Connection pad terminals
          C_1S = C_1_g + sum_{j>=3} C_1_cp_j
          C_2S = C_2_g + sum_{j>=3} C_2_cp_j
          C_sigma = C_12 + (C_1S * C_2S) / (C_1S + C_2S)
    """
    dim = matrix.shape[0]
    if dim >= 3 and matrix.shape[1] >= 3:
        c12 = abs(float(matrix[1, 2]))
        # Sum of capacitances to ground (index 0) and any coupling pads (indices 3..dim-1)
        c1_s = abs(float(matrix[1, 0])) + sum(abs(float(matrix[1, j])) for j in range(3, dim))
        c2_s = abs(float(matrix[2, 0])) + sum(abs(float(matrix[2, j])) for j in range(3, dim))
        g_sum = c1_s + c2_s
        c_g_eff = (c1_s * c2_s) / g_sum if g_sum > 0 else 0.0
        c_sigma = c12 + c_g_eff
        if np.isfinite(c_sigma) and c_sigma > 0:
            return float(c_sigma)

    if dim == 2 and matrix.shape[1] == 2:
        c_mutual = abs(float(matrix[0, 1]))
        if c_mutual > 0:
            return c_mutual
        return abs(float(matrix[1, 1]))

    return abs(float(matrix[0, 0]))


def extract_transmon_hamiltonian_lom(
    matrix: np.ndarray,
    lj_val: float,
    n_connection_pads: int = 1,
    fr_ghz: float = 7.0,
    cj_f: float = 1e-15,
) -> dict[str, Any]:
    """
    Use Qiskit Metal's internal lumped oscillator model (LOM) function
    extract_transmon_coupled_Noscillator to extract full Hamiltonian properties,
    coupling rates (g), dispersive shift (chi), charge dispersion, and Purcell T1.
    """
    try:
        from qiskit_metal.analyses.quantization.lumped_capacitive import (
            extract_transmon_coupled_Noscillator,
        )
    except ImportError as exc:
        raise ImportError("Qiskit Metal is required for LOM Hamiltonian extraction") from exc

    dim = matrix.shape[0]
    if dim < n_connection_pads + 3:
        n_connection_pads = max(0, dim - 3)

    if n_connection_pads < 1:
        c_sigma = extract_transmon_c_sigma(matrix)
        ec_mhz = (E_CHARGE**2 / (2.0 * c_sigma * H_PLANCK)) * 1e-6
        ej_mhz = (PHI_0**2 / (4.0 * np.pi**2 * lj_val * H_PLANCK)) * 1e-6
        fq_ghz = (np.sqrt(8.0 * ej_mhz * ec_mhz) - ec_mhz) / 1000.0
        return {
            "fQ_GHz": float(fq_ghz),
            "EC_MHz": float(ec_mhz),
            "EJ_MHz": float(ej_mhz),
            "alpha_MHz": float(ec_mhz),
            "ratio": float(ej_mhz / max(1e-3, ec_mhz)),
            "C_sigma_fF": float(c_sigma * 1e15),
            "g_MHz": 0.0,
            "chi_MHz": 0.0,
            "dispersion_kHz": 0.0,
            "T1_us": 100.0,
        }

    # Ic from linear Josephson inductance
    ic_val = float(PHI_0 / (2.0 * np.pi * lj_val))

    # Run Qiskit Metal's extract_transmon_coupled_Noscillator
    lom_res = extract_transmon_coupled_Noscillator(
        capMatrix=matrix,
        Ic=ic_val,
        CJ=cj_f,
        N=n_connection_pads,
        fb=[],
        fr=float(fr_ghz),
        print_info=False,
    )

    fq_ghz = float(lom_res.get("fQ", 5.0))
    ec_mhz = float(lom_res.get("EC", 300.0))
    ej_ghz = float(lom_res.get("EJ", 15.0))
    ej_mhz = ej_ghz * 1000.0
    alpha = abs(float(lom_res.get("alpha", ec_mhz)))
    disp = float(lom_res.get("dispersion", 0.1))

    gbus = lom_res.get("gbus", [10.0])
    g_mhz = float(gbus[0]) if hasattr(gbus, "__len__") else float(gbus)

    chibus = lom_res.get("chi_in_MHz", [1.0])
    chi_mhz = float(chibus[0]) if hasattr(chibus, "__len__") else float(chibus)

    from .inverse_design import compute_purcell_t1
    t1_us = compute_purcell_t1(fq_ghz, g_mhz, f_r_GHz=fr_ghz)
    c_sigma = extract_transmon_c_sigma(matrix)

    return {
        "fQ_GHz": fq_ghz,
        "EC_MHz": ec_mhz,
        "EJ_MHz": ej_mhz,
        "alpha_MHz": alpha,
        "ratio": float(ej_mhz / max(1e-3, ec_mhz)),
        "C_sigma_fF": float(c_sigma * 1e15),
        "g_MHz": g_mhz,
        "chi_MHz": chi_mhz,
        "dispersion_kHz": disp,
        "T1_us": t1_us,
    }


def _read_terminal_capacitance(path: Path) -> np.ndarray:
    """Read SQDMetal's terminal-C.csv, whose first column is the row index."""
    frame = pd.read_csv(path, index_col=0)
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"No finite capacitance matrix found in {path}")
    if values.shape[0] != values.shape[1]:
        raise ValueError(f"Capacitance matrix in {path} is not square: {values.shape}")
    return values
