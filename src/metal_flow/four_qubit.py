from qiskit_metal.qlibrary.qubits.transmon_pocket_6 import TransmonPocket6

from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

from qiskit_metal.qlibrary.terminations.launchpad_wb_driven import LaunchpadWirebondDriven

from qiskit_metal.qlibrary.tlines.framed_path import RouteFramed

from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee

from qiskit_metal import designs, MetalGUI

design = designs.DesignPlanar()

gui = MetalGUI(design)


p_0 = LaunchpadWirebondDriven(
design,
name='p_0',
options={'lead_length': '50 um',
 'orientation': '-90',
 'pad_height': '80 um',
 'pad_width': '80 um',
 'pos_x': '25um',
 'pos_y': '8500um'},

component_template=None,
)




p_1 = LaunchpadWirebondDriven(
design,
name='p_1',
options={'lead_length': '50 um',
 'orientation': '-90',
 'pad_height': '80 um',
 'pad_width': '80 um',
 'pos_x': '7250um',
 'pos_y': '8500um'},

component_template=None,
)




p_2 = LaunchpadWirebondDriven(
design,
name='p_2',
options={'lead_length': '50 um',
 'orientation': '90',
 'pad_height': '80 um',
 'pad_width': '80 um',
 'pos_x': '25um',
 'pos_y': '2000um'},

component_template=None,
)




p_3 = LaunchpadWirebondDriven(
design,
name='p_3',
options={'lead_length': '50 um',
 'orientation': '90',
 'pad_height': '80 um',
 'pad_width': '80 um',
 'pos_x': '7250um',
 'pos_y': '2000um'},

component_template=None,
)




ctl_0 = CoupledLineTee(
design,
name='ctl_0',
options={'coupling_length': '500um',
 'down_length': '300um',
 'fillet': '100um',
 'orientation': 90,
 'pos_x': '25um',
 'pos_y': '7000um'},

component_template=None,
)




ctl_1 = CoupledLineTee(
design,
name='ctl_1',
options={'coupling_length': '500um',
 'down_length': '300um',
 'fillet': '100um',
 'mirror': True,
 'orientation': -90,
 'pos_x': '7250um',
 'pos_y': '7000um'},

component_template=None,
)




ctl_2 = CoupledLineTee(
design,
name='ctl_2',
options={'coupling_length': '500um',
 'down_length': '300um',
 'fillet': '100um',
 'orientation': 90,
 'pos_x': '25um',
 'pos_y': '3000um'},

component_template=None,
)




ctl_3 = CoupledLineTee(
design,
name='ctl_3',
options={'coupling_length': '500um',
 'down_length': '300um',
 'fillet': '100um',
 'mirror': True,
 'orientation': -90,
 'pos_x': '7250um',
 'pos_y': '3200um'},

component_template=None,
)




cpw_p_0ctl_0 = RouteFramed(
design,
name='cpw_p_0ctl_0',
options={'_actual_length': '0.9499999999999993 '
                   'mm',
 'pin_inputs': {'end_pin': {'component': 'ctl_0',
                            'pin': 'prime_end'},
                'start_pin': {'component': 'p_0',
                              'pin': 'tie'}},
 'trace_gap': 'cpw_gap'},

type='CPW',
)




cpw_ctl_0ctl_2 = RouteFramed(
design,
name='cpw_ctl_0ctl_2',
options={'_actual_length': '3.0 mm',
 'pin_inputs': {'end_pin': {'component': 'ctl_2',
                            'pin': 'prime_end'},
                'start_pin': {'component': 'ctl_0',
                              'pin': 'prime_start'}},
 'trace_gap': 'cpw_gap'},

type='CPW',
)




cpw_ctl_2p_2 = RouteFramed(
design,
name='cpw_ctl_2p_2',
options={'_actual_length': '0.4500000000000002 '
                   'mm',
 'pin_inputs': {'end_pin': {'component': 'p_2',
                            'pin': 'tie'},
                'start_pin': {'component': 'ctl_2',
                              'pin': 'prime_start'}},
 'trace_gap': 'cpw_gap'},

type='CPW',
)




cpw_p_1ctl_1 = RouteFramed(
design,
name='cpw_p_1ctl_1',
options={'_actual_length': '0.9499999999999993 '
                   'mm',
 'pin_inputs': {'end_pin': {'component': 'ctl_1',
                            'pin': 'prime_start'},
                'start_pin': {'component': 'p_1',
                              'pin': 'tie'}},
 'trace_gap': 'cpw_gap'},

type='CPW',
)




cpw_ctl_1ctl_3 = RouteFramed(
design,
name='cpw_ctl_1ctl_3',
options={'_actual_length': '2.8 mm',
 'pin_inputs': {'end_pin': {'component': 'ctl_3',
                            'pin': 'prime_start'},
                'start_pin': {'component': 'ctl_1',
                              'pin': 'prime_end'}},
 'trace_gap': 'cpw_gap'},

type='CPW',
)




cpw_ctl_3p_3 = RouteFramed(
design,
name='cpw_ctl_3p_3',
options={'_actual_length': '0.6500000000000004 '
                   'mm',
 'pin_inputs': {'end_pin': {'component': 'p_3',
                            'pin': 'tie'},
                'start_pin': {'component': 'ctl_3',
                              'pin': 'prime_end'}},
 'trace_gap': 'cpw_gap'},

type='CPW',
)





            # WARNING
#options_connection_pads failed to have a value
q_0 = TransmonPocket6(
design,
name='q_0',
options={'connection_pads': {'coupler_long_pad': {'cpw_extend': '150um',
                                          'cpw_gap': '6um',
                                          'cpw_width': '10um',
                                          'loc_H': -1,
                                          'loc_W': 0,
                                          'pad_cpw_extent': '25um',
                                          'pad_cpw_shift': '0um',
                                          'pad_gap': '35 '
                                                     'um',
                                          'pad_height': '30um',
                                          'pad_width': '70 '
                                                       'um',
                                          'pocket_extent': '5um',
                                          'pocket_rise': '0um'},
                     'coupler_short_pad': {'cpw_extend': '150um',
                                           'cpw_gap': '6um',
                                           'cpw_width': '10um',
                                           'loc_H': 1,
                                           'loc_W': 1,
                                           'pad_cpw_extent': '25um',
                                           'pad_cpw_shift': '0um',
                                           'pad_gap': '30 '
                                                      'um',
                                           'pad_height': '30um',
                                           'pad_width': '70 '
                                                        'um',
                                           'pocket_extent': '5um',
                                           'pocket_rise': '0um'},
                     'resonator_pad': {'cpw_extend': '150um',
                                       'cpw_gap': '6um',
                                       'cpw_width': '10um',
                                       'loc_H': 1,
                                       'loc_W': -1,
                                       'pad_cpw_extent': '25um',
                                       'pad_cpw_shift': '0um',
                                       'pad_gap': '40 '
                                                  'um',
                                       'pad_height': '30um',
                                       'pad_width': '70 '
                                                    'um',
                                       'pocket_extent': '5um',
                                       'pocket_rise': '0um'}},
 'gds_cell_name': 'Chip0725_auto',
 'orientation': '0',
 'pad_gap': '30 um',
 'pad_height': '100 um',
 'pad_width': '500 um',
 'pos_x': '2250um',
 'pos_y': '6500um'}
)





            # WARNING
#options_connection_pads failed to have a value
q_1 = TransmonPocket6(
design,
name='q_1',
options={'connection_pads': {'coupler_long_pad': {'cpw_extend': '150um',
                                          'cpw_gap': '6um',
                                          'cpw_width': '10um',
                                          'loc_H': -1,
                                          'loc_W': 0,
                                          'pad_cpw_extent': '25um',
                                          'pad_cpw_shift': '0um',
                                          'pad_gap': '35 '
                                                     'um',
                                          'pad_height': '30um',
                                          'pad_width': '70um',
                                          'pocket_extent': '5um',
                                          'pocket_rise': '0um'},
                     'coupler_short_pad': {'cpw_extend': '150um',
                                           'cpw_gap': '6um',
                                           'cpw_width': '10um',
                                           'loc_H': 1,
                                           'loc_W': -1,
                                           'pad_cpw_extent': '25um',
                                           'pad_cpw_shift': '0um',
                                           'pad_gap': '30 '
                                                      'um',
                                           'pad_height': '30um',
                                           'pad_width': '70um',
                                           'pocket_extent': '5um',
                                           'pocket_rise': '0um'},
                     'resonator_pad': {'cpw_extend': '150um',
                                       'cpw_gap': '6um',
                                       'cpw_width': '10um',
                                       'loc_H': 1,
                                       'loc_W': 1,
                                       'pad_cpw_extent': '25um',
                                       'pad_cpw_shift': '0um',
                                       'pad_gap': '40 '
                                                  'um',
                                       'pad_height': '30um',
                                       'pad_width': '70um',
                                       'pocket_extent': '5um',
                                       'pocket_rise': '0um'}},
 'gds_cell_name': 'Chip0726_auto',
 'orientation': '0',
 'pad_gap': '20 um',
 'pad_height': '70 um',
 'pad_width': '450 um',
 'pos_x': '5250um',
 'pos_y': '6500um'}
)





            # WARNING
#options_connection_pads failed to have a value
q_2 = TransmonPocket6(
design,
name='q_2',
options={'connection_pads': {'coupler_long_pad': {'cpw_extend': '150um',
                                          'cpw_gap': '6um',
                                          'cpw_width': '10um',
                                          'loc_H': 1,
                                          'loc_W': 0,
                                          'pad_cpw_extent': '25um',
                                          'pad_cpw_shift': '0um',
                                          'pad_gap': '35 '
                                                     'um',
                                          'pad_height': '30um',
                                          'pad_width': '70um',
                                          'pocket_extent': '5um',
                                          'pocket_rise': '0um'},
                     'coupler_short_pad': {'cpw_extend': '150um',
                                           'cpw_gap': '6um',
                                           'cpw_width': '10um',
                                           'loc_H': -1,
                                           'loc_W': 1,
                                           'pad_cpw_extent': '25um',
                                           'pad_cpw_shift': '0um',
                                           'pad_gap': '30 '
                                                      'um',
                                           'pad_height': '30um',
                                           'pad_width': '70um',
                                           'pocket_extent': '5um',
                                           'pocket_rise': '0um'},
                     'resonator_pad': {'cpw_extend': '150um',
                                       'cpw_gap': '6um',
                                       'cpw_width': '10um',
                                       'loc_H': -1,
                                       'loc_W': -1,
                                       'pad_cpw_extent': '25um',
                                       'pad_cpw_shift': '0um',
                                       'pad_gap': '35 '
                                                  'um',
                                       'pad_height': '30um',
                                       'pad_width': '70um',
                                       'pocket_extent': '5um',
                                       'pocket_rise': '0um'}},
 'gds_cell_name': 'Chip0727_auto',
 'orientation': '0',
 'pad_gap': '40 um',
 'pad_height': '100 um',
 'pad_width': '500 um',
 'pos_x': '2250um',
 'pos_y': '3500um'}
)





            # WARNING
#options_connection_pads failed to have a value
q_3 = TransmonPocket6(
design,
name='q_3',
options={'connection_pads': {'coupler_long_pad': {'cpw_extend': '150um',
                                          'cpw_gap': '6um',
                                          'cpw_width': '10um',
                                          'loc_H': 1,
                                          'loc_W': 0,
                                          'pad_cpw_extent': '25um',
                                          'pad_cpw_shift': '0um',
                                          'pad_gap': '35 '
                                                     'um',
                                          'pad_height': '30um',
                                          'pad_width': '70um',
                                          'pocket_extent': '5um',
                                          'pocket_rise': '0um'},
                     'coupler_short_pad': {'cpw_extend': '150um',
                                           'cpw_gap': '6um',
                                           'cpw_width': '10um',
                                           'loc_H': -1,
                                           'loc_W': -1,
                                           'pad_cpw_extent': '25um',
                                           'pad_cpw_shift': '0um',
                                           'pad_gap': '30 '
                                                      'um',
                                           'pad_height': '30um',
                                           'pad_width': '70um',
                                           'pocket_extent': '5um',
                                           'pocket_rise': '0um'},
                     'resonator_pad': {'cpw_extend': '150um',
                                       'cpw_gap': '6um',
                                       'cpw_width': '10um',
                                       'loc_H': -1,
                                       'loc_W': 1,
                                       'pad_cpw_extent': '25um',
                                       'pad_cpw_shift': '0um',
                                       'pad_gap': '35 '
                                                  'um',
                                       'pad_height': '30um',
                                       'pad_width': '70um',
                                       'pocket_extent': '5um',
                                       'pocket_rise': '0um'}},
 'gds_cell_name': 'Chip0728_auto',
 'orientation': '0',
 'pad_gap': '30 um',
 'pad_height': '100 um',
 'pad_width': '500 um',
 'pos_x': '5250um',
 'pos_y': '3500um'}
)




q_0_resonator = RouteMeander(
design,
name='q_0_resonator',
options={'_actual_length': '10.862 mm',
 'fillet': '70um',
 'hfss_wire_bonds': True,
 'lead': {'end_jogged_extension': '',
          'end_straight': '0um',
          'start_jogged_extension': '',
          'start_straight': '100um'},
 'meander': {'asymmetry': '50um',
             'spacing': '150um'},
 'pin_inputs': {'end_pin': {'component': 'q_0',
                            'pin': 'resonator_pad'},
                'start_pin': {'component': 'ctl_0',
                              'pin': 'second_end'}},
 'total_length': '10.862 mm',
 'trace_gap': 'cpw_gap'},

type='CPW',
)




q_1_resonator = RouteMeander(
design,
name='q_1_resonator',
options={'_actual_length': '11.186249999999994 '
                   'mm',
 'fillet': '70um',
 'hfss_wire_bonds': True,
 'lead': {'end_jogged_extension': '',
          'end_straight': '0um',
          'start_jogged_extension': '',
          'start_straight': '200um'},
 'meander': {'asymmetry': '300um',
             'spacing': '150um'},
 'pin_inputs': {'end_pin': {'component': 'q_1',
                            'pin': 'resonator_pad'},
                'start_pin': {'component': 'ctl_1',
                              'pin': 'second_end'}},
 'total_length': '11.18625 mm',
 'trace_gap': 'cpw_gap'},

type='CPW',
)




q_2_resonator = RouteMeander(
design,
name='q_2_resonator',
options={'_actual_length': '10.555999999999996 '
                   'mm',
 'fillet': '70um',
 'hfss_wire_bonds': True,
 'lead': {'end_jogged_extension': '',
          'end_straight': '0um',
          'start_jogged_extension': '',
          'start_straight': '100um'},
 'meander': {'asymmetry': '300um',
             'spacing': '150um'},
 'pin_inputs': {'end_pin': {'component': 'q_2',
                            'pin': 'resonator_pad'},
                'start_pin': {'component': 'ctl_2',
                              'pin': 'second_end'}},
 'total_length': '10.556 mm',
 'trace_gap': 'cpw_gap'},

type='CPW',
)




q_3_resonator = RouteMeander(
design,
name='q_3_resonator',
options={'_actual_length': '10.26675 mm',
 'fillet': '70um',
 'hfss_wire_bonds': True,
 'lead': {'end_jogged_extension': '',
          'end_straight': '0um',
          'start_jogged_extension': '',
          'start_straight': '200um'},
 'meander': {'asymmetry': '-400um',
             'spacing': '150um'},
 'pin_inputs': {'end_pin': {'component': 'q_3',
                            'pin': 'resonator_pad'},
                'start_pin': {'component': 'ctl_3',
                              'pin': 'second_end'}},
 'total_length': '10.26675 mm',
 'trace_gap': 'cpw_gap'},

type='CPW',
)




c_q0_q1 = RouteMeander(
design,
name='c_q0_q1',
options={'_actual_length': '12.491250000000008 '
                   'mm',
 'fillet': '70um',
 'hfss_wire_bonds': True,
 'lead': {'end_jogged_extension': '',
          'end_straight': '100um',
          'start_jogged_extension': '',
          'start_straight': '100um'},
 'meander': {'asymmetry': '300um',
             'spacing': '150um'},
 'pin_inputs': {'end_pin': {'component': 'q_1',
                            'pin': 'coupler_short_pad'},
                'start_pin': {'component': 'q_0',
                              'pin': 'coupler_short_pad'}},
 'total_length': '12.49125 mm',
 'trace_gap': 'cpw_gap'},

type='CPW',
)




c_q2_q3 = RouteMeander(
design,
name='c_q2_q3',
options={'_actual_length': '11.186249999999996 '
                   'mm',
 'fillet': '70um',
 'hfss_wire_bonds': True,
 'lead': {'end_jogged_extension': '',
          'end_straight': '100um',
          'start_jogged_extension': '',
          'start_straight': '100um'},
 'meander': {'asymmetry': '-200um',
             'spacing': '150um'},
 'pin_inputs': {'end_pin': {'component': 'q_3',
                            'pin': 'coupler_short_pad'},
                'start_pin': {'component': 'q_2',
                              'pin': 'coupler_short_pad'}},
 'total_length': '11.18625 mm',
 'trace_gap': 'cpw_gap'},

type='CPW',
)




c_q0_q2 = RouteMeander(
design,
name='c_q0_q2',
options={'_actual_length': '11.530250000000006 '
                   'mm',
 'fillet': '70um',
 'hfss_wire_bonds': True,
 'lead': {'end_jogged_extension': '',
          'end_straight': '200um',
          'start_jogged_extension': '',
          'start_straight': '200um'},
 'meander': {'asymmetry': '400um',
             'spacing': '150um'},
 'pin_inputs': {'end_pin': {'component': 'q_2',
                            'pin': 'coupler_long_pad'},
                'start_pin': {'component': 'q_0',
                              'pin': 'coupler_long_pad'}},
 'total_length': '11.53025 mm',
 'trace_gap': 'cpw_gap'},

type='CPW',
)




c_q1_q3 = RouteMeander(
design,
name='c_q1_q3',
options={'_actual_length': '11.530249999999993 '
                   'mm',
 'fillet': '70um',
 'hfss_wire_bonds': True,
 'lead': {'end_jogged_extension': '',
          'end_straight': '100um',
          'start_jogged_extension': '',
          'start_straight': '100um'},
 'meander': {'asymmetry': '-250um',
             'spacing': '200um'},
 'pin_inputs': {'end_pin': {'component': 'q_3',
                            'pin': 'coupler_long_pad'},
                'start_pin': {'component': 'q_1',
                              'pin': 'coupler_long_pad'}},
 'total_length': '11.53025 mm',
 'trace_gap': 'cpw_gap'},

type='CPW',
)



gui.rebuild()
gui.autoscale()