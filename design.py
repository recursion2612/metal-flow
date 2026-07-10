from qiskit_metal import Dict, view
from functions import create_design


FOUR_QUBIT_DESIGN_DICT = Dict(
    # Define the overall dimensions of the planar silicon chip
    chip_size = Dict(
        size_x = '10mm',
        size_y = '10mm',
        size_z = '-280um',
        centre_x = '4mm',
        centre_y = '5mm'
    ),

    # Define standard Coplanar Waveguide (CPW) parameters used across the chip
    cpw_dims = Dict(
        width = '10 um',
        gap = '6 um'
    ),

    

    # Substrate and metal film thickness for EM simulations (Ansys Q3D/HFSS)
    physical_params = Dict(
        substrate_thickness = 280e-6,
        film_thickness = 200e-9
    ),

    # Locations and orientations of the 4 wirebond launchpads for I/O
    launchpad_options = Dict(
        p0 = Dict(pos_x='250um' , pos_y='8500um', orientation='-90', lead_length='50 um', pad_width='80 um', pad_height='80 um'),
        p1 = Dict(pos_x='7250um', pos_y='8500um', orientation='-90', lead_length='50 um', pad_width='80 um', pad_height='80 um'),
        p2 = Dict(pos_x='250um' , pos_y='2000um' , orientation='90', lead_length='50 um', pad_width='80 um', pad_height='80 um'),
        p3 = Dict(pos_x='7250um', pos_y='2000um' , orientation='90', lead_length='50 um', pad_width='80 um', pad_height='80 um')
    ),


    cpw_default_options = Dict(chip='main', fillet='99.99um', hfss_wire_bonds=True),


    feedline_connections = [('p0', 'p2', 'feedline_02'), ('p1', 'p3', 'feedline_13')],


    # Locations, orientations, and GDS generation names for the 4 Transmon qubits
    qubit_options = Dict(
        q0 = Dict(pos_x='2250um', pos_y='6000um', orientation='0', gds_cell_name='Chip0725_auto', pad_width= '450 um', pad_height= "100 um", pad_gap = '20 um'),#, pocket_width='1100 um', pocket_height='1100 um'),
        q1 = Dict(pos_x='5250um', pos_y='6000um', orientation='0', gds_cell_name='Chip0726_auto', pad_width= '450 um', pad_height= "70 um" , pad_gap = '20 um'), #, pocket_width='1100 um', pocket_height='1100 um'),
        q2 = Dict(pos_x='2250um', pos_y='3000um', orientation='0', gds_cell_name='Chip0727_auto', pad_width= '450 um', pad_height= "90 um" , pad_gap = '20 um'), #, pocket_width='1100 um', pocket_height='1100 um'),
        q3 = Dict(pos_x='5250um', pos_y='3000um', orientation='0', gds_cell_name='Chip0728_auto', pad_width= '450 um', pad_height= "80 um" , pad_gap = '20 um')  #, pocket_width='1100 um', pocket_height='1100 um')
    ),

    # Default geometry parameters for the TransmonPocket6 components, including coupling pads
    transmon_defaults = Dict(
        connection_pads=dict(
            resonator_pad = dict(pad_width='70um', cpw_extend='150um', pad_gap='30um'),
            coupler_short_pad = dict(pad_width='70um', cpw_extend='150um', pad_gap='30um'),
            coupler_long_pad = dict(pad_width='70um', cpw_extend='150um', pad_gap='30um')
        )
    ),

    # Explicitly defining the pad placement (Width/Height directional vectors) for each qubit
    # to ensure routing doesn't cross over itself
    qubit_pad_placements = Dict(
        q0 = dict(resonator_pad=dict(loc_W=-1, loc_H=+1), coupler_short_pad=dict(loc_W=+1, loc_H=+1), coupler_long_pad=dict(loc_W=0, loc_H=-1)),
        q1 = dict(resonator_pad=dict(loc_W=+1, loc_H=+1), coupler_short_pad=dict(loc_W=-1, loc_H=+1), coupler_long_pad=dict(loc_W=0, loc_H=-1)),
        q2 = dict(resonator_pad=dict(loc_W=-1, loc_H=-1), coupler_short_pad=dict(loc_W=+1, loc_H=-1), coupler_long_pad=dict(loc_W=0, loc_H=+1)),
        q3 = dict(resonator_pad=dict(loc_W=+1, loc_H=-1), coupler_short_pad=dict(loc_W=-1, loc_H=-1), coupler_long_pad=dict(loc_W=0, loc_H=+1)),
    ),

    # Target qubit frequencies 
    qubit_frequencies_hz = Dict(
        q0 = 5.2,
        q1 = 5.5, 
        q2 = 6.1, 
        q3 = 5.8
    ),

    # Target bare frequencies for the readout resonators
    resonator_frequencies_ghz = Dict(
        q0 = 6.2,
        q1 = 6.7,
        q2 = 7.1,
        q3 = 7.3
    ),

    # Meander routing configuration for readout resonators (lead lengths, fillet radii, asymmetry)
    resonator_meander_params = Dict(
        q0 = Dict(asymmetry = -400, lead=Dict(start_straight='50um', end_straight='50um')),
        q1 = Dict(asymmetry = 400,  lead=Dict(start_straight='50um', end_straight='50um')),
        q2 = Dict(asymmetry = 400,  lead=Dict(start_straight='900um', end_straight='50um')),
        q3 = Dict(asymmetry = -400, lead=Dict(start_straight='600um', end_straight='300um'))
    ),

    # Defined total lengths for CPW bus couplers between qubits
    coupler_frequencies_ghz = Dict(
        c01 = 6.0,
        c02 = 6.9,
        c13 = 6.5,
        c23 = 6.7
    ),

    # Meander routing configuration for inter-qubit CPW couplers
    coupler_meander_params = Dict(
        c01 = Dict(asymmetry = 300,   lead=Dict(start_straight='100um', end_straight='100um')),
        c02 = Dict(asymmetry = 525.5, lead=Dict(start_straight='200um', end_straight='200um')),
        c13 = Dict(asymmetry = -300,  lead=Dict(start_straight='100um', end_straight='50um')),
        c23 = Dict(asymmetry = -300,  lead=Dict(start_straight='100um', end_straight='100um'))
        ),

    otg_list = [('otg0', dict(chip='main', pos_x='300um' , pos_y='6500um', orientation=270)),
                ('otg1', dict(chip='main', pos_x='7200um', pos_y='6500um', orientation=270)), 
                ('otg2', dict(chip='main', pos_x='300um' , pos_y='4500um', orientation=90 )),
                ('otg3', dict(chip='main', pos_x='7200um', pos_y='4500um', orientation=90))],

    readout_map = {'otg0':'q0', 'otg1':'q1', 'otg2':'q2', 'otg3':'q3'}, 

    coupling_map = dict(
        c01=('q0', 'coupler_short_pad', 'q1', 'coupler_short_pad'),
        c23=('q2', 'coupler_short_pad', 'q3', 'coupler_short_pad'),
        c02=('q0', 'coupler_long_pad' , 'q2',  'coupler_long_pad'),
        c13=('q1', 'coupler_long_pad' , 'q3',  'coupler_long_pad')
    ),

    bus_freq = dict(q0 = [6.0, 6.9], q1=[6.0, 6.5], q2=[6.7, 6.9], q3=[6.7,6.5])
    
)




if __name__ == '__main__':
    design = create_design(FOUR_QUBIT_DESIGN_DICT)
    view(design).savefig("design.png")
