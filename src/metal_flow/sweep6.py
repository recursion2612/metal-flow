# Cell 1 — Imports + design skeleton
from qiskit_metal import designs, draw, MetalGUI, Dict
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.pathfinder import RoutePathfinder



from qiskit_metal.analyses import LOManalysis

from itertools import product

import pandas as pd

design = designs.DesignPlanar()
design.overwrite_enabled = True

design.chips.main.size['size_x'] = '11mm'
design.chips.main.size['size_y'] = '11mm'
# Cell 2 (updated) — Q0–Q3 unchanged (TransmonPocket, 3 pads each fits the 4-corner limit).
# Q4 rebuilt as TransmonPocket6 with 5 angularly-placed connection pads.

from qiskit_metal.qlibrary.qubits.transmon_pocket_6 import TransmonPocket6


arm = "2 mm"


pad_opts = Dict(pad_width='50um', pad_height='30um', pad_gap='30um')

q0 = TransmonPocket6(design, 'Q0', options=Dict(
    pos_x=f'-{arm}', pos_y=f'{arm}', orientation=90, pad_width='425um', pocket_height='650um',
    connection_pads=Dict(
        to_Q4  =Dict(loc_W=-1, loc_H=-1, **pad_opts),
        to_Q2  =Dict(loc_W=-1, loc_H=+1, **pad_opts),
        readout=Dict(loc_W=+1, loc_H=+1, **pad_opts),
    )))

design.rebuild()


lom = LOManalysis(design, 'q3d')

qubit_name = 'Q0'

sweep_setup = Dict(design = design,
                        # Only simulate q0 and define its open terminations to extract exact capacitance matrix
                        run_args_dict = Dict(components=[design.components[qubit_name].name] , open_terminations=[(design.components[qubit_name].name, pad) for pad in design.components[qubit_name].pins.keys()], box_plus_buffer=True),
                        # LOM physics constraints (Josephson Junction Inductance & Capacitance)
                        lom_setup = Dict(junctions=Dict(Lj=10, Cj=2), freq_readout=[7.0], freq_bus=[6.0, 6.2]),
                        # Ansys Q3D Iterative solver properties
                        lom_sim_setup = Dict(name = f'{qubit_name}_setup', 
                                             reuse_selected_design = True, 
                                             reuse_setup = True,
                                             freq_ghz = 5.0,
                                             save_fields = True, 
                                             enabled = True, 
                                             max_passes = 35, 
                                             min_passes = 2, 
                                             min_converged_passes = 2, 
                                             percent_error = 0.5, 
                                             percent_refinement = 30, 
                                             auto_increase_solution_order = True, 
                                             solution_order = 'Highest', 
                                             solver_type = 'Iterative',
                                             ))



print(sweep_setup)

pad_widths = [width for width in range(400, 600, 25)]
pad_heights = [height for height in range(50, 200, 15)]
pad_gaps = [gap for gap in range(10, 55, 5)]

pin_pad_widths  = [width for width in range(30, 200, 20)]
pin_pad_heights = [height for height in range(20, 50, 5)]
pin_pad_gaps = [gap for gap in range(5, 50, 5)]

Ljs = [Lj for Lj in range(5, 26, 1)]

lom.setup = sweep_setup.lom_setup
lom.sim.setup = sweep_setup.lom_sim_setup

data_list = []


# Define the parameter grids
qubit_grid = product(pad_widths, pad_heights, pad_gaps)
pin_grid = product(pin_pad_widths, pin_pad_heights, pin_pad_gaps)

# Combine both grids to iterate through every configuration
for q_params, p_params in product(qubit_grid, pin_grid):
    w, h, g = q_params
    pw, ph, pg = p_params
    
    # Update main qubit pads
    qubit = design.components[qubit_name]
    qubit.options.pad_width = f"{w}um"
    qubit.options.pad_height = f"{h}um"
    qubit.options.pad_gap = f"{g}um"
    
    # Update all connection pads
    for pin in qubit.pin_names:
        qubit.options.connection_pads[pin].pad_width = f"{pw}um"
        qubit.options.connection_pads[pin].pad_height = f"{ph}um"
        qubit.options.connection_pads[pin].pad_gap = f"{pg}um"
        
        design.rebuild()
        result_dict = {"pad_width_um":w, "pad_height_um":h, "pad_gap_um":g, "pin_pad_width_um":pw, "pin_pad_height_um":ph, "pin_pad_gap_um":pg}
        print("="*10+"Iteration"+"="*10)
        print(result_dict)

        lom.sim.design.rebuild()

        lom.sim.run(name=qubit_name+'_capacitive', **sweep_setup.run_args_dict)

        for inductance in Ljs:
            lom.setup.junctions.Lj = inductance
            lom.run_lom()
            result_dict.update({"input_Lj":inductance})
            result_dict.update(lom.lumped_oscillator)
            data_list.append(result_dict)



df = pd.DataFrame(data_list)

df.to_csv("data6.csv")

lom.sim.close()
del lom


        


