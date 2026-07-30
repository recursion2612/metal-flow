from qiskit_metal import Dict, view
from metal_flow.functions import create_design


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
        p_0 = Dict(pos_x='25um' , pos_y='8500um', orientation='-90', lead_length='50 um', pad_width='80 um', pad_height='80 um'),
        p_1 = Dict(pos_x='7250um', pos_y='8500um', orientation='-90', lead_length='50 um', pad_width='80 um', pad_height='80 um'),
        p_2 = Dict(pos_x='25um' , pos_y='2000um' , orientation='90', lead_length='50 um', pad_width='80 um', pad_height='80 um'),
        p_3 = Dict(pos_x='7250um', pos_y='2000um' , orientation='90', lead_length='50 um', pad_width='80 um', pad_height='80 um')
    ),


    cpw_default_options = Dict(chip='main', hfss_wire_bonds=True),


    feedline_connections = [Dict(start_pin=Dict(component='p_0', pin='tie'), end_pin=Dict(component='ctl_0', pin='prime_end')), 
                            
                            Dict(start_pin=Dict(component='ctl_0', pin='prime_start'), end_pin=Dict(component='ctl_2', pin='prime_end')),
                            
                            Dict(start_pin=Dict(component='ctl_2', pin='prime_start'), end_pin=Dict(component='p_2', pin='tie')),
                            
                            Dict(start_pin=Dict(component='p_1', pin='tie'), end_pin=Dict(component='ctl_1', pin='prime_start')), 
                            
                            Dict(start_pin=Dict(component='ctl_1', pin='prime_end'), end_pin=Dict(component='ctl_3', pin='prime_start')),
                            
                            Dict(start_pin=Dict(component='ctl_3', pin='prime_end'), end_pin=Dict(component='p_3', pin='tie'))

    ],


    # Locations, orientations, and GDS generation names for the 4 Transmon qubits
    qubit_options = Dict(
        q_0 = Dict(pos_x='2250um', pos_y='6500um', orientation='0', gds_cell_name='Chip0725_auto', pad_width= '450 um', pad_height= "100 um", pad_gap = '20 um'),#, pocket_width='1100 um', pocket_height='1100 um'),
        q_1 = Dict(pos_x='5250um', pos_y='6500um', orientation='0', gds_cell_name='Chip0726_auto', pad_width= '450 um', pad_height= "70 um" , pad_gap = '20 um'), #, pocket_width='1100 um', pocket_height='1100 um'),
        q_2 = Dict(pos_x='2250um', pos_y='3500um', orientation='0', gds_cell_name='Chip0727_auto', pad_width= '450 um', pad_height= "90 um" , pad_gap = '20 um'), #, pocket_width='1100 um', pocket_height='1100 um'),
        q_3 = Dict(pos_x='5250um', pos_y='3500um', orientation='0', gds_cell_name='Chip0728_auto', pad_width= '450 um', pad_height= "80 um" , pad_gap = '20 um')  #, pocket_width='1100 um', pocket_height='1100 um')
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
        q_0 = Dict(resonator_pad=Dict(loc_W=-1, loc_H=+1), coupler_short_pad=Dict(loc_W=+1, loc_H=+1), coupler_long_pad=Dict(loc_W=0, loc_H=-1)),
        q_1 = Dict(resonator_pad=Dict(loc_W=+1, loc_H=+1), coupler_short_pad=Dict(loc_W=-1, loc_H=+1), coupler_long_pad=Dict(loc_W=0, loc_H=-1)),
        q_2 = Dict(resonator_pad=Dict(loc_W=-1, loc_H=-1), coupler_short_pad=Dict(loc_W=+1, loc_H=-1), coupler_long_pad=Dict(loc_W=0, loc_H=+1)),
        q_3 = Dict(resonator_pad=Dict(loc_W=+1, loc_H=-1), coupler_short_pad=Dict(loc_W=-1, loc_H=-1), coupler_long_pad=Dict(loc_W=0, loc_H=+1)),
    ),

    # Target qubit frequencies 
    qubit_frequencies_hz = Dict(
        q_0 = 5.2,
        q_1 = 5.5, 
        q_2 = 6.1, 
        q_3 = 5.8
    ),

    # Target bare frequencies for the readout resonators
    resonator_frequencies_ghz = Dict(
        q_0 = 6.2,
        q_1 = 6.7,
        q_2 = 7.1,
        q_3 = 7.3
    ),

    # Meander routing configuration for readout resonators (lead lengths, fillet radii, asymmetry)
    resonator_meander_params = Dict(
        q_0 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '50um'), lead=Dict(start_straight='100um', end_straight='0um')),
        q_1 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '300um'), lead=Dict(start_straight='200um', end_straight='0um')),
        q_2 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '300um'), lead=Dict(start_straight='100um', end_straight='0um')),
        q_3 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '-400um'), lead=Dict(start_straight='200um', end_straight='0um'))
    ),

    # Defined total lengths for CPW bus couplers between qubits
    coupler_frequencies_ghz = Dict(
        c_q0_q1 = 6.0,
        c_q0_q2 = 6.9,
        c_q1_q3 = 6.5,
        c_q2_q3 = 6.7
    ),

    # Meander routing configuration for inter-qubit CPW couplers
    coupler_meander_params = Dict(
        c_q0_q1 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '300um'), lead=Dict(start_straight='100um', end_straight='100um')),
        c_q0_q2 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '400um'), lead=Dict(start_straight='200um', end_straight='200um')),
        c_q1_q3 = Dict(fillet='70um', meander=Dict(spacing = '200um', asymmetry = '-250um'), lead=Dict(start_straight='100um', end_straight='100um')),
        c_q2_q3 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '-200um'), lead=Dict(start_straight='100um', end_straight='100um'))
        ),

    
    readout_map = [Dict(start_pin= Dict(component='ctl_0', pin='second_end'), end_pin=Dict(component='q_0', pin='resonator_pad')), 
                   Dict(start_pin= Dict(component='ctl_1', pin='second_end'), end_pin=Dict(component='q_1', pin='resonator_pad')), 
                   Dict(start_pin= Dict(component='ctl_2', pin='second_end'), end_pin=Dict(component='q_2', pin='resonator_pad')), 
                   Dict(start_pin= Dict(component='ctl_3', pin='second_end'), end_pin=Dict(component='q_3', pin='resonator_pad')), 
                   ],

    coupled_tline_params = Dict(
                                ctl_0= Dict(pos_x='25um'  , pos_y='7000um', open_termination=True,  orientation=90 , down_length='300um', coupling_length='500um', fillet='100um', mirror=False),
                                ctl_1= Dict(pos_x='7250um', pos_y='7000um', open_termination=True,  orientation=-90, down_length='300um', coupling_length='500um', fillet='100um', mirror=True),
                                ctl_2= Dict(pos_x='25um'  , pos_y='3000um', open_termination=True,  orientation=90 , down_length='300um', coupling_length='500um', fillet='100um', mirror=False),
                                ctl_3= Dict(pos_x='7250um', pos_y='3200um', open_termination=True,  orientation=-90, down_length='300um', coupling_length='500um', fillet='100um', mirror=True)
                                ),

    coupling_map = dict(
        c_q0_q1=('q_0', 'coupler_short_pad', 'q_1', 'coupler_short_pad'),
        c_q2_q3=('q_2', 'coupler_short_pad', 'q_3', 'coupler_short_pad'),
        c_q0_q2=('q_0', 'coupler_long_pad' , 'q_2',  'coupler_long_pad'),
        c_q1_q3=('q_1', 'coupler_long_pad' , 'q_3',  'coupler_long_pad')
    ),

    bus_freq = dict(q_0 = [6.0, 6.9], q_1=[6.0, 6.5], q_2=[6.7, 6.9], q_3=[6.7,6.5])
    
)



# FIVE_QUBIT_DESIGN_DICT = Dict(
#     # Define the overall dimensions of the planar silicon chip
#     chip_size = Dict(
#         size_x = '10mm',
#         size_y = '10mm',
#         size_z = '-280u`m',
#         centre_x = '4mm',
#         centre_y = '5mm'
#     ),

#     # Define standard Coplanar Waveguide (CPW) parameters used across the chip
#     cpw_dims = Dict(
#         width = '10 um',
#         gap = '6 um'
#     ),

    

#     # Substrate and metal film thickness for EM simulations (Ansys Q3D/HFSS)
#     physical_params = Dict(
#         substrate_thickness = 280e-6,
#         film_thickness = 200e-9
#     ),

#     # Locations and orientations of the 4 wirebond launchpads for I/O
#     launchpad_options = Dict(
#         p0 = Dict(pos_x='25um' , pos_y='8500um', orientation='-90', lead_length='50 um', pad_width='80 um', pad_height='80 um'),
#         p1 = Dict(pos_x='7250um', pos_y='8500um', orientation='-90', lead_length='50 um', pad_width='80 um', pad_height='80 um'),
#         p2 = Dict(pos_x='25um' , pos_y='2000um' , orientation='90', lead_length='50 um', pad_width='80 um', pad_height='80 um'),
#         p3 = Dict(pos_x='7250um', pos_y='2000um' , orientation='90', lead_length='50 um', pad_width='80 um', pad_height='80 um')
#     ),


#     cpw_default_options = Dict(chip='main', hfss_wire_bonds=True),


#     feedline_connections = [Dict(start_pin=Dict(component='p0', pin='tie'), end_pin=Dict(component='ctl0', pin='prime_end')), 
                            
#                             Dict(start_pin=Dict(component='ctl0', pin='prime_start'), end_pin=Dict(component='ctl2', pin='prime_end')),
                            
#                             Dict(start_pin=Dict(component='ctl2', pin='prime_start'), end_pin=Dict(component='p2', pin='tie')),
                            
#                             Dict(start_pin=Dict(component='p1', pin='tie'), end_pin=Dict(component='ctl1', pin='prime_start')), 
                            
#                             Dict(start_pin=Dict(component='ctl1', pin='prime_end'), end_pin=Dict(component='ctl3', pin='prime_start')),
                            
#                             Dict(start_pin=Dict(component='ctl3', pin='prime_end'), end_pin=Dict(component='p3', pin='tie'))

#     ],


#     # Locations, orientations, and GDS generation names for the 4 Transmon qubits
#     qubit_options = Dict(
#         q0 = Dict(pos_x='1500um', pos_y='1500um', orientation='0', gds_cell_name='Chip0725_auto', pad_width= '450 um', pad_height= "100 um", pad_gap = '20 um'),#, pocket_width='1100 um', pocket_height='1100 um'),
#         q1 = Dict(pos_x='1500um', pos_y='3000um', orientation='0', gds_cell_name='Chip0726_auto', pad_width= '450 um', pad_height= "70 um" , pad_gap = '20 um'), #, pocket_width='1100 um', pocket_height='1100 um'),
#         q2 = Dict(pos_x='3000um', pos_y='4500um', orientation='0', gds_cell_name='Chip0727_auto', pad_width= '450 um', pad_height= "90 um" , pad_gap = '20 um'), #, pocket_width='1100 um', pocket_height='1100 um'),
#         q3 = Dict(pos_x='5250um', pos_y='3500um', orientation='0', gds_cell_name='Chip0728_auto', pad_width= '450 um', pad_height= "80 um" , pad_gap = '20 um')  #, pocket_width='1100 um', pocket_height='1100 um')
#         q4 = Dict(pos_x='5250um', pos_y='3500um', orientation='0', gds_cell_name='Chip0729_auto', pad_width= '450 um', pad_height= "80 um" , pad_gap = '20 um')  #, pocket_width='1100 um', pocket_height='1100 um')
#     ),

#     # Default geometry parameters for the TransmonPocket6 components, including coupling pads
#     transmon_defaults = Dict(
#         connection_pads=dict(
#             resonator_pad = dict(pad_width='70um', cpw_extend='150um', pad_gap='30um'),
#             coupler_short_pad = dict(pad_width='70um', cpw_extend='150um', pad_gap='30um'),
#             coupler_long_pad = dict(pad_width='70um', cpw_extend='150um', pad_gap='30um')
#         )
#     ),

#     # Explicitly defining the pad placement (Width/Height directional vectors) for each qubit
#     # to ensure routing doesn't cross over itself
#     qubit_pad_placements = Dict(
#         q0 = Dict(resonator_pad=Dict(loc_W=-1, loc_H=+1), coupler_short_pad=Dict(loc_W=+1, loc_H=+1), coupler_long_pad=Dict(loc_W=0, loc_H=-1)),
#         q1 = Dict(resonator_pad=Dict(loc_W=+1, loc_H=+1), coupler_short_pad=Dict(loc_W=-1, loc_H=+1), coupler_long_pad=Dict(loc_W=0, loc_H=-1)),
#         q2 = Dict(resonator_pad=Dict(loc_W=-1, loc_H=-1), coupler_short_pad=Dict(loc_W=+1, loc_H=-1), coupler_long_pad=Dict(loc_W=0, loc_H=+1)),
#         q3 = Dict(resonator_pad=Dict(loc_W=+1, loc_H=-1), coupler_short_pad=Dict(loc_W=-1, loc_H=-1), coupler_long_pad=Dict(loc_W=0, loc_H=+1)),
#     ),

#     # Target qubit frequencies 
#     qubit_frequencies_hz = Dict(
#         q0 = 5.2,
#         q1 = 5.5, 
#         q2 = 6.1, 
#         q3 = 5.8
#     ),

#     # Target bare frequencies for the readout resonators
#     resonator_frequencies_ghz = Dict(
#         q0 = 6.2,
#         q1 = 6.7,
#         q2 = 7.1,
#         q3 = 7.3
#     ),

#     # Meander routing configuration for readout resonators (lead lengths, fillet radii, asymmetry)
#     resonator_meander_params = Dict(
#         q0 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '50um'), lead=Dict(start_straight='100um', end_straight='0um')),
#         q1 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '300um'), lead=Dict(start_straight='200um', end_straight='0um')),
#         q2 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '300um'), lead=Dict(start_straight='100um', end_straight='0um')),
#         q3 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '-400um'), lead=Dict(start_straight='200um', end_straight='0um'))
#     ),

#     # Defined total lengths for CPW bus couplers between qubits
#     coupler_frequencies_ghz = Dict(
#         c01 = 6.0,
#         c02 = 6.9,
#         c13 = 6.5,
#         c23 = 6.7
#     ),

#     # Meander routing configuration for inter-qubit CPW couplers
#     coupler_meander_params = Dict(
#         c01 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '300um'), lead=Dict(start_straight='100um', end_straight='100um')),
#         c02 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '400um'), lead=Dict(start_straight='200um', end_straight='200um')),
#         c13 = Dict(fillet='70um', meander=Dict(spacing = '200um', asymmetry = '-250um'), lead=Dict(start_straight='100um', end_straight='100um')),
#         c23 = Dict(fillet='70um', meander=Dict(spacing = '150um', asymmetry = '-200um'), lead=Dict(start_straight='100um', end_straight='100um'))
#         ),

    
#     readout_map = [Dict(start_pin= Dict(component='ctl0', pin='second_end'), end_pin=Dict(component='q0', pin='resonator_pad')), 
#                    Dict(start_pin= Dict(component='ctl1', pin='second_end'), end_pin=Dict(component='q1', pin='resonator_pad')), 
#                    Dict(start_pin= Dict(component='ctl2', pin='second_end'), end_pin=Dict(component='q2', pin='resonator_pad')), 
#                    Dict(start_pin= Dict(component='ctl3', pin='second_end'), end_pin=Dict(component='q3', pin='resonator_pad')), 
#                    ],

#     coupled_tline_params = Dict(
#                                 ctl0= Dict(pos_x='25um'  , pos_y='7000um', open_termination=True,  orientation=90 , down_length='300um', coupling_length='500um', fillet='100um', mirror=False),
#                                 ctl1= Dict(pos_x='7250um', pos_y='7000um', open_termination=True,  orientation=-90, down_length='300um', coupling_length='500um', fillet='100um', mirror=True),
#                                 ctl2= Dict(pos_x='25um'  , pos_y='3000um', open_termination=True,  orientation=90 , down_length='300um', coupling_length='500um', fillet='100um', mirror=False),
#                                 ctl3= Dict(pos_x='7250um', pos_y='3200um', open_termination=True,  orientation=-90, down_length='300um', coupling_length='500um', fillet='100um', mirror=True)
#                                 ),

#     coupling_map = dict(
#         c01=('q0', 'coupler_short_pad', 'q1', 'coupler_short_pad'),
#         c23=('q2', 'coupler_short_pad', 'q3', 'coupler_short_pad'),
#         c02=('q0', 'coupler_long_pad' , 'q2',  'coupler_long_pad'),
#         c13=('q1', 'coupler_long_pad' , 'q3',  'coupler_long_pad')
#     ),

#     bus_freq = dict(q0 = [6.0, 6.9], q1=[6.0, 6.5], q2=[6.7, 6.9], q3=[6.7,6.5])
    
#     )




if __name__ == '__main__':
    # from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
    # from qiskit_metal import designs
    # design = designs.DesignPlanar()
    # ctl = CoupledLineTee(design)

    # print(ctl.pin_names)
    design = create_design(FOUR_QUBIT_DESIGN_DICT)
    view(design).savefig("design.png")
