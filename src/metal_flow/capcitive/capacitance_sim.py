import sys, logging
import pandas as pd
from qiskit_metal import Dict

import h5py

## Analysis
from qiskit_metal.analyses.quantization import LOManalysis


class Capacitence_Sweeper():
    """Custom class to handle automated sweeping of component variables and cap_cap_lom data extraction.
       sweep_setup should contain something like this
       - lom_setup = Dict(junctions=Dict(Lj=12, Cj=2), freq_readout=7.0,freq_bus=[6.0, 6.2])
       - lom_sim_setup = Dict({'name': sweep_setup.name+'_Setup',
                              'reuse_selected_design': True, 
                              'reuse_setup': True, 
                              'freq_ghz': 5.0, 
                              'save_fields': True, 
                              'enabled': True, 
                              'max_passes': 15, 
                              'min_passes': 2, 
                              'min_converged_passes': 2, 
                              'percent_error': 0.5, 
                              'percent_refinement': 30, 
                              'auto_increase_solution_order': True, 
                              'solution_order': 'Highest', 
                              'solver_type': 'Iterative'})
       - run_args_dict = Dict(components = [], open_terminations = [], box_plus_buffer = True/False)
    """
        
    def __init__(self, sweep_setup):
        # Initialize LOM analysis binding it to 'q3d'
        self.lom = LOManalysis(sweep_setup.design, 'q3d') 
        
        # Inject user setups or fallback to defaults
        self.lom.setup = sweep_setup.lom_setup if sweep_setup.lom_setup else Dict(junctions=Dict(Lj=12, Cj=2), freq_readout=7.0,freq_bus=[6.0, 6.2])
        self.lom.sim.setup = sweep_setup.lom_sim_setup if sweep_setup.lom_sim_setup else Dict({'name': sweep_setup.name+'_Setup', 'reuse_selected_design': True, 'reuse_setup': True, 'freq_ghz': 5.0, 'save_fields': True, 'enabled': True, 'max_passes': 15, 'min_passes': 2, 'min_converged_passes': 2, 'percent_error': 0.5, 'percent_refinement': 30, 'auto_increase_solution_order': True, 'solution_order': 'Highest', 'solver_type': 'Iterative'})
        self.run_args_dict = sweep_setup.run_args_dict # Contains args for _render/analyse

        # self.name = sweep_setup.name if sweep_setup.name else 'Sweeper_Q3D'
        # self.textfile_str  = sweep_setup.name+'_text_log' if name else 'f{self.name}_text_log'
        # self.imagefile_str = sweep_setup.name+'_image' if name else 'f{self.name}_image'


    def perform_initial_analysis(self):

        '''This function shall perform an initial capacitance analysis.
           Goal is to extract initial values and then perform sweep on any one of the parameters.
        '''
        # Ensure modifications if any are made to the design
        self.lom.sim.design.rebuild()

        self.lom.sim.run(components = self.run_args_dict.components, open_terminations = self.run_args_dict.open_terminations, box_plus_buffer=True)
        
        self.lom.run_lom()

        lom_result = self.lom.lumped_oscillator
        all_lom_df = self.lom.lumped_oscillator_all

        print(f"Is converged:{self.lom.sim.is_converged}\nTotal passes:{len(self.lom.sim.capacitance_all_passes)}")

        plot_freq_alpha_conv = self.lom.plot_convergence()
        plot_chi_coup = self.lom.plot_convergence_chi()

        self.lom.sim.close()

        return lom_result, all_lom_df, plot_freq_alpha_conv, plot_chi_coup

        

    @staticmethod
    def _sanitize_for_hdf5(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert any object-dtype columns (strings, mixed types, units-as-str, etc.)
        into fixed-length byte strings so h5py/to_records can serialize them.
        Leaves numeric/bool columns untouched.
        """
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == object:
                # str() first handles any stray None/np.nan/non-str objects safely
                df[col] = df[col].astype(str).astype('S')
        return df


    def perform_sweep(self, name, sweep_component_options, sweep_variable, sweep_values, track_parameters=None):
        """
        Executes the parameter sweep, logging progress to console alongside
        pyaedt/Qiskit Metal/pyEPR logs, and saving results incrementally to CSV.
        """

        h5_path = f"{name}_data.h5"

        print(f"{sweep_variable} begins ({len(sweep_variable)} values)")

        with h5py.File(h5_path, 'w') as h5file:

            for value in sweep_values:
                self.lom.clear_data()

                iter_group = h5file.create_group(f"{sweep_variable}_{value}")

                print(f"ITERATION {sweep_variable} = {value}")

                sweep_component_options[sweep_variable] = value
                self.lom.sim.design.rebuild()
                self.lom.sim.run(
                    components=self.run_args_dict.components,
                    open_terminations=self.run_args_dict.open_terminations,
                )
                self.lom.run_lom()

                data_dict = {
                    sweep_variable: value, # value is str 
                    # 'cap_matrix': self.lom.sim.capacitance_matrix,
                }

                if track_parameters:
                    for parameter in track_parameters:
                        data_dict[parameter] = self.lom.lumped_oscillator[parameter]
                else:
                    data_dict.update(self.lom.lumped_oscillator)

                data_dict.update({'is_converged': self.lom.sim.is_converged, 'passes': len(self.lom.sim.capacitance_all_passes)})

                data_df = pd.DataFrame(data_dict) # sweep_variable dtype is object

                iter_group.create_dataset("capacitance_matrix", data=self.lom.sim.capacitance_matrix, compression="gzip")

                cap_mat_all_pass_group = iter_group.create_group("cap_mat_all_passes")

                for iter, mat in self.lom.sim.capacitance_all_passes.items():
                    cap_mat_all_pass_group.create_dataset(str(iter), data=mat, compression="gzip")

                iter_group.create_dataset('iter_details', data=self._sanitize_for_hdf5(data_df).to_records(index=False))

                iter_group.create_dataset('lom_details', data=self._sanitize_for_hdf5(self.lom.lumped_oscillator_all).to_records(index=False))

                print(f"ITERATION {sweep_variable}_{value} done | converged={data_dict['is_converged']} | passes={data_dict['passes']}")


            print(f"{sweep_variable} iterations complete, results in {h5_path}")

            self.lom.sim.close()

        return "Sweep Completed"
    

if __name__ == '__main__':
    from metal_flow.design import FOUR_QUBIT_DESIGN_DICT


# class SweepGeneral():
#     '''Class to generate sweeps across multiple parameter values. Purpose is to build a pandas database for the sweeps and have functions to plot the data.
#         Main job 
#         1. Configure and run sim for different parameter values. If data already exists, skip. This should happen with no breaks.
#         2. Save data as pandas df.
#         3. Have functions to plot 
#     '''

#     def __init__(self, design, simulation, file_prefix):
#         self.simulation = simulation(design)
#         self.file_prefix = file_prefix
        
#     def run_sweep(parameters, values):


        