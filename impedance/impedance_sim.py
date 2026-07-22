from qiskit_metal.analyses import ScatteringImpedanceSim
from qiskit_metal import Dict


class Sweeper_SYZ():

    def __init__(self, solution_type, config_setup):
        self.setup= config_setup
        self.solution_type = solution_type

    def extract_syz(self, design, design_name, selection, open_pins, port_list, box_plus_buffer):

        em = ScatteringImpedanceSim(design, self.solution_type)
        
        em.setup.name = f"{design_name}_setup"
        ## Test setup 
        em.setup.freq_ghz = 6.0  # Try to keep this at the center of the swept frequency range for 'fast' sweeps and at the largest frequency for interpolating sweep for the best results
        em.setup.max_delta_s = 0.1 #0.005  # This is necessary to get good results if interpolating sweep is not working for you
        em.setup.max_passes = 10
        em.setup.min_passes = 2
        em.setup.basis_order = 1 # -1  # Mixed order

        print(em.setup)
        
        em.setup.sweep_setup.name = f"{design_name}_sweep"
        em.setup.sweep_setup.start_ghz = 2.0
        em.setup.sweep_setup.stop_ghz = 10.0
        em.setup.sweep_setup.count = 101 # 10001
        em.setup.sweep_setup.type = "Interpolating"
        
        print(em.setup.sweep_setup)

        em.renderer.start()
        em.setup.renderer.options['x_buffer_width_mm'] = 0.1
        em.setup.renderer.options['y_buffer_width_mm'] = 0.1
        
        em._render(name = design_name,
                    selection=selection,
                    solution_type="drivenmodal",
                    vars_to_initialize=em.setup.vars,
                    open_pins=open_pins,
                    port_list=port_list,
                    box_plus_buffer=box_plus_buffer)



        em._analyze()

        freqs, Pcurves, Pparams = em.renderer.get_params([f"S{i+1}1" for i in range(len(port_list))])

        conv_t, conv_f, text = em.renderer.get_convergences()

        return Pparams, conv_t, conv_f, text
    


if __name__ == '__main__':
    from design import FOUR_QUBIT_DESIGN_DICT
    from functions import create_design
    test_design = create_design(FOUR_QUBIT_DESIGN_DICT)

    SYZ=Dict(solution_type='hfss',
                       setup=Dict(freq_ghz=6,
                                  max_delta_s=0.1,
                                  basis_order=1,
                                  min_passes=1,
                                  max_passes=10,
                                  min_converged=1,
                                  pct_refinement=30,
                                  vars=Dict(Lj='10 nH', Cj='0 fF'),
                                  sweep_setup=Dict(name= 'Sweep',
                                                   start_ghz= 2.0,
                                                    stop_ghz= 10.0,
                                                    count= 1001,
                                                    type= 'Interpolating',
                                                    save_fields= False
                                                  )
                                 )
            )

    Sweeper = Sweeper_SYZ(**SYZ)

    Pparams, conv_t, conv_f, text = Sweeper.extract_syz(design=test_design, design_name="feedline_02", 
                                                  selection=['p0', 'p2', 'feedline_02'], 
                                                  open_pins=[], 
                                                  port_list=[('p0', 'in', 50), ('p2', 'in', 50)], 
                                                  box_plus_buffer=True)
    

    
    




