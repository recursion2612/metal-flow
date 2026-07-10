import sys, logging
import pandas as pd
from qiskit_metal import Dict

## Analysis
from qiskit_metal.analyses.quantization import LOManalysis


class Sweeper():
    """Custom class to handle automated sweeping of component variables and cap_cap_lom data extraction."""
        
    def __init__(self, sweep_setup):
        # Initialize LOM analysis binding it to 'q3d'
        self.lom = LOManalysis(sweep_setup.design, 'q3d') 
        
        # Inject user setups or fallback to defaults
        self.lom.setup = sweep_setup.lom_setup if sweep_setup.lom_setup else Dict(junctions=Dict(Lj=12, Cj=2), freq_readout=7.0,freq_bus=[6.0, 6.2])
        self.lom.sim.setup = sweep_setup.lom_sim_setup if sweep_setup.lom_sim_setup else Dict({'name': sweep_setup.name+'_Setup', 'reuse_selected_design': True, 'reuse_setup': True, 'freq_ghz': 5.0, 'save_fields': True, 'enabled': True, 'max_passes': 15, 'min_passes': 2, 'min_converged_passes': 2, 'percent_error': 0.5, 'percent_refinement': 30, 'auto_increase_solution_order': True, 'solution_order': 'Highest', 'solver_type': 'Iterative'})
        self.run_args_dict = sweep_setup.run_args_dict
        # self.name = sweep_setup.name if sweep_setup.name else 'Sweeper_Q3D'
        # self.textfile_str  = sweep_setup.name+'_text_log' if name else 'f{self.name}_text_log'
        # self.imagefile_str = sweep_setup.name+'_image' if name else 'f{self.name}_image'


    def perform_sweep(self, name, sweep_component_options, sweep_variable, sweep_values, track_parameters=None):
        """
        Executes the parameter sweep. 
        Hooks into the standard sys and metal loggers to redirect voluminous Ansys logs to a text file 
        instead of flooding the Jupyter notebook/console.
        """
        # Save original output streams to restore later
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Hook into Qiskit Metal's logging
        metal_logger = logging.getLogger('qiskit_metal')
        
        with open(f'{name}_log.txt', 'a', encoding='utf-8') as loggerfile:
            # Redirect standard outputs to log file
            sys.stdout = loggerfile
            sys.stderr = loggerfile

            # Detach existing handlers to prevent console double-logging
            old_handlers = metal_logger.handlers[:]
            for handler in old_handlers:
                metal_logger.removeHandler(handler)
                
            # Attach custom file handler
            file_handler = logging.StreamHandler(loggerfile)
            metal_logger.addHandler(file_handler)

            try:
                iteration = 0
                data_list = []
                
                for value in sweep_values:
                    # Clear simulation memory before new iteration
                    self.lom.clear_data()

                    if iteration == 0:
                        print("|"*50 + f"{sweep_variable} Begins" +"|"*50+'\n')

                    # Mutate the design parameter
                    sweep_component_options[sweep_variable] = value

                    print("="*50 + f"ITERATION {iteration}" + "="*50+'\n')
                    print(sweep_variable+f" = {value}\n")

                    # Rebuild design in Metal to register the geometric change
                    self.lom.sim.design.rebuild()
                    
                    # Push the single component geometry to Ansys Q3D and execute extraction
                    self.lom.sim.run(components = self.run_args_dict.components, open_terminations = self.run_args_dict.open_terminations)
                    

                    # Run Energy Participation/LOM math on the capacitance matrix to get physics parameters
                    self.lom.run_lom()

                    # Save iteration data
                    data_dict = {'Iteration':iteration, 'pad_gap':sweep_component_options.pad_gap, 'pad_width':sweep_component_options.pad_width, 'pad_height':sweep_component_options.pad_height, 'cap_matrix':self.lom.sim.capacitance_matrix}
                    
                    if track_parameters:
                        for parameter in track_parameters:
                            data_dict[parameter] = self.lom.lumped_oscillator[parameter]
                    else:
                        
                        data_dict.update(self.lom.lumped_oscillator)


                    data_dict.update({'cap_matrix_all_passes':self.lom.sim.capacitance_all_passes, 'lom_all_passes':self.lom.lumped_oscillator_all, 'is_converged':self.lom.sim.is_converged, 'passes':len(self.lom.sim.capacitance_all_passes)})
                    data_list.append(data_dict)
                    iteration += 1

                # Export tracked LOM data to CSV using pandas
                data_df = pd.DataFrame(data_list)
                data_df.to_csv(f"{name}_data.csv", index=False)
                del data_df

            finally:
                # CRITICAL: Always restore stdout/stderr even if Ansys crashes mid-sweep
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                metal_logger.removeHandler(file_handler)
                for handler in old_handlers:
                    metal_logger.addHandler(handler)

        return "Sweep Completed"
    

class SweepGeneral():
    '''Class to generate sweeps across multiple parameter values. Purpose is to build a pandas database for the sweeps and have functions to plot the data.
        Main job 
        1. Configure and run sim for different parameter values. If data already exists, skip. This should happen with no breaks.
        2. Save data as pandas df.
        3. Have functions to plot 
    '''

    def __init__(self, design, simulation, file_prefix):
        self.simulation = simulation(design)
        self.file_prefix = file_prefix
        
    def run_sweep(parameters, values):


        