"""
Quantum Chip Parameter Optimization Library
Modules for AWS Palace interfacing, PhysicsNeMo surrogate modeling, and real-valued GA optimization.
"""

from .palace_cad_interface import (
    update_qiskit_geometry,
    export_to_gmsh,
    create_palace_config,
    run_palace_solver,
    parse_palace_results,
)
from .nemo_surrogate import PhysicsNeMoSurrogate
from .ga_optimizer import (
    compute_cost,
    select_parents,
    crossover_sbx,
    mutate_polynomial,
    produce_next_generation,
)
from .cad_adapter import (
    SqdmetalCapacitanceRunner,
    create_transmon_design,
    export_qiskit_metal_gmsh,
)

__all__ = [
    "update_qiskit_geometry",
    "export_to_gmsh",
    "create_palace_config",
    "run_palace_solver",
    "parse_palace_results",
    "PhysicsNeMoSurrogate",
    "compute_cost",
    "select_parents",
    "crossover_sbx",
    "mutate_polynomial",
    "produce_next_generation",
    "SqdmetalCapacitanceRunner",
    "create_transmon_design",
    "export_qiskit_metal_gmsh",
]
