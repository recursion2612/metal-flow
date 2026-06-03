from qiskit_metal.analyses import ScatteringImpedanceSim


def extract_syz(design, design_name, selection, open_pins, port_list, box_plus_buffer):
    em = ScatteringImpedanceSim(design, "hfss")


    em.setup.name = f"{design_name}-setup"
    em.setup.freq_ghz = 6.0  # Try to keep this at the center of the swept frequency range for 'fast' sweeps and at the largest frequency for interpolating sweep for the best results
    em.setup.max_delta_s = 0.005  # This is necessary to get good results if interpolating sweep is not working for you
    em.setup.max_passes = 20
    em.setup.min_passes = 2
    em.setup.basis_order = -1  # Mixed order

    print(em.setup)
    
    em.setup.sweep_setup.name = f"{design_name}-sweep"
    em.setup.sweep_setup.start_ghz = 2.0
    em.setup.sweep_setup.stop_ghz = 10.0
    em.setup.sweep_setup.count = 10001
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
    

