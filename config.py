from qiskit_metal import Dict

config = Dict(Capacitive=Dict(solution_type='q3d',
                              setup=Dict(name = f'q3d_setup',
                                         reuse_selected_design = True,
                                         reuse_setup = True,
                                         freq_ghz = 5.0,
                                         save_fields = True,
                                         enabled = True,
                                         max_passes = 30,
                                         min_passes = 2,
                                         min_converged_passes = 3,
                                         percent_error = 0.05,
                                         percent_refinement = 30,
                                         auto_increase_solution_order = True,
                                         solution_order = 'Highest',
                                         solver_type = 'Iterative'
                                         )
                              ),
              Eigenmode=Dict(solution_type='hfss',
                             max_passes = 30,
                             max_delta_f = 0.1,
                             min_freq_ghz = 1.1,
                             n_modes=1,
                             pct_refinement=30,

                             ),
              SYZ=Dict(solution_type='hfss',
                       setup=Dict(freq_ghz=6,
                                  max_delta_s=0.001,
                                  basis_order=-1,
                                  min_passes=1,
                                  max_passes=35,
                                  min_converged=2,
                                  pct_refinement=30,
                                  vars=Dict(Lj='10 nH', Cj='0 fF'),
                                  sweep_setup=Dict(name= 'Sweep',
                                                   start_ghz= 2.0,
                                                    stop_ghz= 10.0,
                                                    count= 10001,
                                                    type= 'Interpolating',
                                                    save_fields= True
                                                  )
                                )
                       )
               )


if __name__ == '__main__':
    print(config)