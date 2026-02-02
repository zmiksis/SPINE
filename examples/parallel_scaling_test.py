# parallel_scaling_test.py
# Zachary M. Miksis
# November 2025

# Example script to test parallel scaling of the implemented solver.
# See calcium_wave_synapse.py for a full description of the model being
# simulated here.

# May need to run with increased open file limits on some systems:
#   ulimit -n 2048

# AMD Ryzen Threadripper PRO 5955WX:
#   1 process: ~1945 seconds
#   2 processes: ~961 seconds
#   4 processes: ~497 seconds
#   8 processes: ~257 seconds
#   16 processes: ~144 seconds
#   32 processes: ~127 seconds

# Import module for generating neuron model
from spine.utils import NeuronModel
# Import solver 
from spine.solver import SBDF
# Import graph generation functions
from spine.graph import gen_beam
# Import synapse class
from spine.synapse import synapse

# Import additional libraries as needed
import time
import argparse
import copy

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--proc", type=int)
    parser.add_argument("--neurons", type=int)
    args = parser.parse_args()

    # Setting number of processes
    if args.proc is not None:
        proc = args.proc
    else:
        proc = 1

    # Setting number of neurons
    if args.neurons is not None:
        num_neurons = args.neurons
    else:
        num_neurons = 128

    # Create neuron model instance
    neuron = NeuronModel()

    # Top data directory, where problem setup and results are stored
    neuron.settings.top_folder = './'

    # Subdirectory with data for this specific problem
    neuron.settings.data_folder = 'parallel_scaling/'
    neuron.settings.output_folder = 'parallel_scaling-output/'

    # Settings for coupled calcium model
    neuron.model['CYT_buffer'] = True
    neuron.model['PM_leak'] = True
    neuron.model['PMCA'] = True
    neuron.model['NCX'] = True
    neuron.model['SERCA'] = True
    neuron.model['ER_leak'] = True
    neuron.model['RyR'] = True
    neuron.model['IP3'] = True
    neuron.model['synapse'] = True

    # Turn on voltage coupling and ionic currents
    neuron.model['voltage_coupling'] = True
    neuron.currents['I_Na'] = True
    neuron.currents['I_K'] = True
    neuron.currents['I_Leak'] = True
    # Couple calcium and electrical activity
    neuron.currents['I_Ca'] = True

    # Record all cytosolic calcium data
    neuron.recorder['full'] = True
    neuron.recorder['full_er'] = True
    neuron.recorder['full_voltage'] = True

    # Create beam neuron
    neuron.geom_builder['const_rad'] = True
    neuron.settings.rad = 0.2 # um
    gen_beam(neuron, 64, 2501) # um length, 1001 nodes

    # Final time (s)
    neuron.settings.T = 60.e-3 # seconds
    # Fixed time step
    neuron.settings.set_time_step(5.e-6) # seconds

    # Stimulate neuron at node 0
    syn = synapse()
    syn.type = 'constant'
    syn.duration = 1.e-3 # seconds
    syn.j = 2.5e-12
    syn.node = [500]

    # IP3 influx at node 0
    syn_ip3 = synapse()
    syn_ip3.type = 'linear'
    syn_ip3.duration = 200.e-3 # seconds
    syn_ip3.j = 5.0e-12
    syn_ip3.node = [500]
    syn_ip3.domain = 'ip3'

    # Attach list of synapse instances to first neuron
    neuron.synapse_instances = [syn, syn_ip3]

    # Create list of neurons for parallel run
    # The first neuron is the original, others are copies
    neuronList = [neuron]
    for _ in range(num_neurons - 1):
        neuronCopy = copy.deepcopy(neuron)
        neuronCopy.settings.output_folder = f'parallel_test-output-{len(neuronList)}/'
        # Turn off recording for all but first neuron to save space
        neuronCopy.recorder['full'] = False
        neuronCopy.recorder['full_er'] = False
        neuronCopy.recorder['full_voltage'] = False
        neuronList.append(neuronCopy)

    # Run solver
    # Time the solver
    time0 = time.time()
    SBDF(neuronList, n_workers=proc, threads_per_process=1)
    time1 = time.time()

    print(f"Time taken for solver: {time1 - time0} seconds")

if __name__ == "__main__":
    main()
