
from qiskit_metal import Dict
from qiskit_metal import designs
# Importing specific Qiskit Metal components for a planar superconducting quantum chip
from qiskit_metal.qlibrary.qubits.transmon_pocket_6 import TransmonPocket6
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.pathfinder import RoutePathfinder
from qiskit_metal.qlibrary.terminations.launchpad_wb_coupled import LaunchpadWirebondCoupled
from qiskit_metal.qlibrary.terminations.launchpad_wb_driven import LaunchpadWirebondDriven
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.toolbox_metal.parsing import parse_value

#CPW calculations
from qiskit_metal.analyses.em.cpw_calculations import guided_wavelength


def connect(design, cpw_name: str, pin1_comp_name: str, pin1_comp_pin: str, pin2_comp_name: str, pin2_comp_pin: str,
            length: str, cpw_options: Dict, asymmetry='0 um'):
    """
    Helper function to abstract the creation of a RouteMeander object connecting two pins.
    Constructs the routing pin_inputs dictionary and sets meander length/asymmetry.
    """
    myoptions = Dict(pin_inputs=Dict(start_pin=Dict(component=pin1_comp_name,pin=pin1_comp_pin),end_pin=Dict(component=pin2_comp_name,pin=pin2_comp_pin)),
        total_length=length)
    myoptions.update(cpw_options)
    myoptions.meander.asymmetry = asymmetry
    return RouteMeander(design, cpw_name, myoptions)

def cal_cpw_wavelength_dict(cpw_freq, resonator_pad, substrate_thickness, film_thickness):
    """
    Uses Qiskit Metal's built-in CPW calculators to determine the required guided wavelength 
    (lambda) for a given frequency, based on line width/gap and dielectric thicknesses.
    Returns the required lambda/4 length string needed to generate a quarter-wavelength resonator.
    """
    vars_ = Dict({'x':5.0, 'y':'5um', 'cpw_width':'10um'})
    parsed_line_width = parse_value(resonator_pad.pad_width, vars_)
    parsed_line_gap = parse_value(resonator_pad.pad_gap, vars_)


    cpw_freq = cpw_freq * (10**9) # Converting Ghz to hz
    resonator_dict = {'freq':cpw_freq}
    
    # Extract lambda, effective epsilon, and quality factor limit
    (resonator_dict['lambda'], resonator_dict['eps_eff'], resonator_dict['qf']) = guided_wavelength(
        freq=cpw_freq, 
        line_width=parsed_line_width, 
        line_gap=parsed_line_gap, 
        substrate_thickness=substrate_thickness, 
        film_thickness=film_thickness
    )
    
    # Format and store the lambda/4 length for direct insertion into RouteMeander
    resonator_dict['lambda'] = round(1000*resonator_dict['lambda'], 3)
    resonator_dict['lambda4_str'] = str(resonator_dict['lambda']/4) + ' mm'
    return resonator_dict



def create_design(DESIGN_DICT):
    # Create a planar design object and instantiate the GUI
    design = designs.DesignPlanar()

    design.overwrite_enabled = True # Allows overriding components with the same name during testing

    # --- Set Global Chip Properties ---
    design.variables['cpw_width'] = DESIGN_DICT.cpw_dims.width
    design.variables['cpw_gap'] = DESIGN_DICT.cpw_dims.gap
    design._chips['main']['size']['size_x'] = DESIGN_DICT.chip_size.size_x
    design._chips['main']['size']['size_y'] = DESIGN_DICT.chip_size.size_y
    design._chips['main']['size']['center_x'] = DESIGN_DICT.chip_size.centre_x
    design._chips['main']['size']['center_y'] = DESIGN_DICT.chip_size.centre_y
    design._chips['main']['size']['size_z'] = DESIGN_DICT.chip_size.size_z
    # --- Loop-Based Component Generation ---
    print("Generating Launchpads...")
    for name, options in DESIGN_DICT.launchpad_options.items():
        LaunchpadWirebondDriven(design, name, options=options)

    print("Generating Feedlines")

    feedline_dict = {}
    for p_start, p_end, f_name in DESIGN_DICT.feedline_connections:

        fline_options = dict(pin_inputs=Dict(start_pin=Dict(component=p_start, pin='tie'), end_pin=Dict(component=p_end, pin='tie')))

        fline_options.update(DESIGN_DICT.cpw_default_options)
        feedline_dict[f_name] =  RoutePathfinder(design, f_name, options=fline_options)

    

    print("Generating Qubits...")
    qubit_components = {}
    for name, options in DESIGN_DICT.qubit_options.items():
        full_opts = DESIGN_DICT.transmon_defaults.copy()
        
        # Apply specific width/height mapping to ensure pins align correctly for routing
        for pad_name, placement in DESIGN_DICT.qubit_pad_placements[name].items():
            full_opts.connection_pads[pad_name].update(placement)
            
        qubit_components[name] = TransmonPocket6(design, name, options=dict(**options, **full_opts))


    print("Generating Open to Ground for resonators")

    for otg, otg_opt in DESIGN_DICT.otg_list:
        OpenToGround(design, otg, options=otg_opt)

    print("Generating Readout Resonators...")
    
    for otg_name, q_name in DESIGN_DICT.readout_map.items():
        # Fetch parameters to dynamically compute lambda/4 length for exact frequency
        resonator_pad = qubit_components[q_name].options.connection_pads.resonator_pad

        resonator_dict = cal_cpw_wavelength_dict(DESIGN_DICT.resonator_frequencies_ghz[q_name], resonator_pad, **DESIGN_DICT.physical_params)
        meander_params = DESIGN_DICT.resonator_meander_params[q_name]
        
        cpw_options = Dict(asymmetry=f'{meander_params.asymmetry}um', lead=meander_params.lead)

        cpw_options.update(DESIGN_DICT.cpw_default_options)

        cpw_options.update(total_length=resonator_dict['lambda4_str'], pin_inputs=Dict(start_pin=Dict(component=otg_name, pin='open'), end_pin=Dict(component=q_name, pin='resonator_pad')))
        
        # Route the resonator using the dynamically calculated quarter-wavelength string
        RouteMeander(design, f'{q_name}_resonator', options=cpw_options)
        

    print("Generating Couplers...")
    
    for coup_name in DESIGN_DICT.coupling_map.keys():
        meander_params = DESIGN_DICT.coupler_meander_params[coup_name]
        
        q1_name = DESIGN_DICT.coupling_map[coup_name][0]
        q1_pad  = DESIGN_DICT.coupling_map[coup_name][1]
        q2_name = DESIGN_DICT.coupling_map[coup_name][2]
        q2_pad  = DESIGN_DICT.coupling_map[coup_name][3]

        coupler_pad = qubit_components[DESIGN_DICT.coupling_map[coup_name][0]].options.connection_pads[q1_pad]

        coupler_dict = cal_cpw_wavelength_dict(DESIGN_DICT.coupler_frequencies_ghz[coup_name], coupler_pad, **DESIGN_DICT.physical_params)

        cpw_options = Dict(asymmetry=f'{meander_params.asymmetry}um', lead=meander_params.lead)

        cpw_options.update(DESIGN_DICT.cpw_default_options)

        cpw_options.update(total_length=coupler_dict['lambda4_str'], pin_inputs=Dict(start_pin=Dict(component=q1_name, pin=q1_pad), end_pin=Dict(component=q2_name, pin=q2_pad)))
        
        # Route the coupler using the dynamically calculated quarter-wavelength string
        RouteMeander(design, coup_name, options=cpw_options)

    return design
