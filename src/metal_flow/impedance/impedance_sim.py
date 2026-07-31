from qiskit_metal.analyses import ScatteringImpedanceSim
from qiskit_metal import Dict


class Sweeper_SYZ():

    def __init__(self, solution_type, setup):
        self.setup = setup
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

        result_dict = Dict(S=Dict(Params=None, fig=None), Y=Dict(Params=None, fig=None), Z=Dict(Params=None, fig=None), convergence=Dict(conv_t=None, conv_f=None, text=None))

        result_dict.S.Params, result_dict.S.fig = em.renderer.plot_params([f'S{i}1' for i in range(1, len(port_list)+1)] )

        result_dict.Z.Params, result_dict.Z.fig = em.renderer.plot_params([f'Z{i}1' for i in range(1, len(port_list)+1)] )

        result_dict.Y.Params, result_dict.Y.fig = em.renderer.plot_params([f'Y{i}1' for i in range(1, len(port_list)+1)] )

        result_dict.convergence.conv_t, result_dict.convergence.conv_f, result_dict.convergence.text = em.renderer.get_convergences()

        em.close()

        return result_dict
    


if __name__ == '__main__':
    from metal_flow.design import FOUR_QUBIT_DESIGN_DICT
    from metal_flow.design import create_design
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

    result_dict = Sweeper.extract_syz(design=test_design, design_name="Hanger_resonators02", 
                                                  selection=['p0', 'p2', 'q0_resonator', 'q2_resonator', 'ctl0', 'ctl2', 'cpw_p0ctl0', 'cpw_ctl0ctl2', 'cpw_ctl2p2'], 
                                                  open_pins=[('q0_resonator', 'end'), ('q2_resonator', 'end')], 
                                                  port_list=[('p0', 'in', 50), ('p2', 'in', 50)], 
                                                  box_plus_buffer=True)
    
    result_dict.S.fig.savefig("test-s-param.png")
    result_dict.Z.fig.savefig("test-z-param.png")
    result_dict.Y.fig.savefig("test-y-param.png")

    result_dict.S.Params.to_csv("test-s-param.csv")
    result_dict.Z.Params.to_csv("test-z-param.csv")
    result_dict.Y.Params.to_csv("test-y-param.csv")
    result_dict.convergence.conv_t.to_csv("test-conv-t.csv")
    result_dict.convergence.conv_f.to_csv("test-conv-f.csv")

    print(result_dict.convergence.text)




    
    




