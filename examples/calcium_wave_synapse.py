# calcium_wave_synapse.py
# Zachary M. Miksis
# November 2025

# Example script to simulate a calcium wave in a neuron beam with
# synaptic coupling to a second neuron beam. The first neuron is
# stimulated with calcium and IP3 influx at the midpoint, generating
# a calcium wave that propagates to the synapse at the end of the neuron. The
# synapse releases calcium into the postsynaptic neuron based on the
# presynaptic cytosolic calcium concentration, generating a calcium wave.
# The first neuron does not have voltage coupling, while the second neuron
# does, along with example AMPA synaptic current. The first neuron is given 
# a precomputed voltage trace loaded from file. The whole simulation runs
# in parallel using up to two processes (one per neuron).

# WARNING: Running this script may return the following warnings:
#
#   RuntimeWarning: Mean of empty slice.
#   RuntimeWarning: invalid value encountered in scalar divide
#
# This is fine and can be ignored. It occurs because the solver is attempting to
# compute the mean calcium concentration in the soma, and the soma does not exist
# in this example. A beam is generated as a dendrite without a soma compartment.

# NOTE: All ion channel and exchange mechanism parameters can be manually set in 
# this script if desired by accessing the relevant attributes of the neuron object. 
# We use default biological parameters here for simplicity.

# Stable calcium wave setup has been adapted from:
#   P. Borole, "CalciumSim: Simulator for calcium dynamics on neuron graphs
#   using dimensionally reduced model," MS thesis, Temple University, 2022.

# Import module for generating neuron model
from spine.utils import NeuronModel
# Import solver 
from spine.solver import SBDF
# Import graph generation functions
from spine.graph import gen_beam
# Import synapse class
from spine.synapse import synapse
# Import synapse communicator
from spine.synapse import SynapseCommunicator
# Import problem settings to customize voltage reading
from spine.utils.settings import ProblemSettings

# Import additional libraries as needed
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import argparse

# Function to read voltage from precomputed voltage trace
def custom_voltage_reader(self, *args, **kwargs):
    tid = kwargs.get('tid', None)
    length = kwargs.get('length', None)
    V0 = self.volt_scale*self.volt_vector[tid,1]*np.ones(length)
    return V0

# Function defining synaptic release profile based on presynaptic calcium
def release_profile(self, *args, **kwargs):
    comm = kwargs.get('comm', None)
    comm_iter = kwargs.get('comm_iter', None)
    node = kwargs.get('node', None)
    neuron = kwargs.get('neuron', None)
    C, comm_node = comm.get_pre_C0(comm_iter)
    if node not in comm_node:
        return 0.0
    return 1e3 * (1.e-15*C[comm_node.index(node)] - neuron.settings.cceq) # uM/s influx proportional to presynaptic cytosolic calcium

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--proc", type=int)
    args = parser.parse_args()

    # Setting number of processes
    if args.proc is not None:
        proc = args.proc
    else:
        proc = 1

    # Create neuron model instance
    neuron = NeuronModel()

    # Customize voltage reading function
    # Do this before changing other settings, as it creates a new ProblemSettings instance
    neuron.settings = ProblemSettings(read_voltage=custom_voltage_reader)
    neuron.settings.volt_vector = np.loadtxt('volt.txt')

    # Top data directory, where problem setup and results are stored
    neuron.settings.top_folder = './'

    # Subdirectory with data for this specific problem
    neuron.settings.data_folder = 'calcium_wave_synapse/'
    neuron.settings.output_folder = 'calcium_wave_synapse-output/'

    # Settings for coupled calcium model
    neuron.model['CYT_buffer'] = True
    neuron.model['PMCA'] = True
    neuron.model['NCX'] = True
    neuron.model['PM_leak'] = True
    neuron.model['SERCA'] = True
    neuron.model['ER_leak'] = True
    neuron.model['RyR'] = True
    neuron.model['IP3'] = True
    neuron.model['synapse'] = True

    # Record all cytosolic calcium data
    neuron.recorder['full'] = True
    neuron.recorder['full_er'] = True
    neuron.recorder['full_voltage'] = True

    # Create beam neuron
    neuron.geom_builder['const_rad'] = True
    neuron.settings.rad = 0.2 # um
    gen_beam(neuron, 64, 1001) # um length, 1001 nodes

    # Final time (s)
    neuron.settings.T = 60.e-3 # seconds
    # Fixed time step
    neuron.settings.set_time_step(5.e-6) # seconds

    # Create second neuron model as a copy of the first
    neuron_copy = copy.deepcopy(neuron)
    neuron.settings.data_folder = 'calcium_wave_synapse/'
    neuron_copy.settings.output_folder = 'calcium_wave_synapse-copy_output/'

    # Turn on voltage coupling and ionic currents
    neuron_copy.model['voltage_coupling'] = True
    neuron_copy.currents['I_Na'] = True
    neuron_copy.currents['I_K'] = True
    neuron_copy.currents['I_Leak'] = True
    # Couple calcium and electrical activity
    neuron_copy.currents['I_Ca'] = True
    # Add example AMPA synaptic current
    neuron_copy.currents['I_syn'] = True

    # Calcium influx at midpoint of first neuron
    syn = synapse()
    syn.type = 'linear'
    syn.duration = 1.e-3 # seconds
    syn.j = 2.5e-12
    syn.node = [500]
    syn.domain = 'cytosol'

    # IP3 influx at midpoint of first neuron
    syn_ip3 = synapse()
    syn_ip3.type = 'linear'
    syn_ip3.duration = 200.e-3 # seconds
    syn_ip3.j = 5.0e-12
    syn_ip3.node = [500]
    syn_ip3.domain = 'ip3'

    # Attach list of synapse instances to first neuron
    neuron.synapse_instances = [syn, syn_ip3]

    # Communication synapse from end of first neuron to start of second neuron
    syn_comm = synapse(release_profile=release_profile)
    syn_comm.domain = 'receptor'
    syn_comm.duration = neuron.settings.T  # entire simulation
    syn_comm.node = [1000]  # last node of second neuron
    neuron_copy.synapse_instances = [syn_comm]

    # Create list of neurons
    neuronList = [neuron, neuron_copy]

    # Create synapse communicator
    communicator = SynapseCommunicator(neuronList)

    # Create synapse connection parameters for voltage coupling
    receptors_per_synapse = 20
    synapses_per_cluster = 20
    neurons_per_cluster = 1
    density = 3
    total_receptors = receptors_per_synapse \
                    * synapses_per_cluster \
                    * neurons_per_cluster \
                    * density

    # Add synapse connection from neuron 0, node 0 to neuron copy, end node
    communicator.add_synapse_connection((0, 0), (1, 1000), total_receptors)

    # Run solver
    time0 = time.time()
    SBDF(neuronList, comm=communicator, n_workers=proc, threads_per_process=1)
    time1 = time.time()

    print(f"Time taken for solver: {time1 - time0} seconds")

    # Load results and plot
    C1 = np.loadtxt('results/' + neuron.settings.output_folder + 'cyt.txt')
    CE1 = np.loadtxt('results/' + neuron.settings.output_folder + 'er.txt')
    V1 = np.loadtxt('results/' + neuron.settings.output_folder + 'volt.txt')

    C2 = np.loadtxt('results/' + neuron_copy.settings.output_folder + 'cyt.txt')
    CE2 = np.loadtxt('results/' + neuron_copy.settings.output_folder + 'er.txt')
    V2 = np.loadtxt('results/' + neuron_copy.settings.output_folder + 'volt.txt')

    x = np.linspace(0, 64, C1.shape[1])
    t = np.linspace(0, 60, C1.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)
    im1 = axes[0].imshow(np.flipud(C1.T),
                        extent=[t[0], t[-1], x[0], x[-1]],
                        aspect='auto',
                        cmap='jet', vmin=0.0, vmax=7.0)
    axes[0].set_title('Presynaptic Neuron')
    axes[0].set_ylabel('Position ($\mu$m)')
    im2 = axes[1].imshow(np.flipud(C2.T),
                        extent=[t[0], t[-1], x[0], x[-1]],
                        aspect='auto',
                        cmap='jet', vmin=0.0, vmax=7.0)
    axes[1].set_title('Postsynaptic Neuron')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Position ($\mu$m)')
    fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.046, pad=0.04, label='$\mu$M')
    fig.suptitle('$Ca^{2+}_{cyt}$', fontsize=14)
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)
    im1 = axes[0].imshow(np.flipud(CE1.T),
                        extent=[t[0], t[-1], x[0], x[-1]],
                        aspect='auto',
                        cmap='jet', vmin=0.0, vmax=275.0)
    axes[0].set_title('Presynaptic Neuron')
    axes[0].set_ylabel('Position ($\mu$m)')
    im2 = axes[1].imshow(np.flipud(CE2.T),
                        extent=[t[0], t[-1], x[0], x[-1]],
                        aspect='auto',
                        cmap='jet', vmin=0.0, vmax=275.0)
    axes[1].set_title('Postsynaptic Neuron')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Position ($\mu$m)')
    fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.046, pad=0.04, label='$\mu$M')
    fig.suptitle('$Ca^{2+}_{er}$', fontsize=14)
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)
    im1 = axes[0].imshow(np.flipud(V1.T),
                        extent=[t[0], t[-1], x[0], x[-1]],
                        aspect='auto',
                        cmap='jet', vmin=-80.0, vmax=40.0)
    axes[0].set_title('Presynaptic Neuron')
    axes[0].set_ylabel('Position ($\mu$m)')
    im2 = axes[1].imshow(np.flipud(V2.T),
                        extent=[t[0], t[-1], x[0], x[-1]],
                        aspect='auto',
                        cmap='jet', vmin=-80.0, vmax=40.0)
    axes[1].set_title('Postsynaptic Neuron')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Position ($\mu$m)')
    fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.046, pad=0.04, label='mV')
    fig.suptitle('$V$', fontsize=14)
    plt.show()

    fig, axes = plt.subplots(1, 1, figsize=(8, 7), sharex=True, sharey=True)
    V1 = V1[6550:6751,:]
    V2 = V2[6550:6751,:]
    t = t[6550:6751]
    im1 = axes.imshow(np.flipud(V2.T),
                        extent=[t[0], t[-1], x[0], x[-1]],
                        aspect='auto',
                        cmap='jet', vmin=-80.0, vmax=40.0)
    axes.set_title('Postsynaptic Neuron')
    axes.set_xlabel('Time (ms)')
    axes.set_ylabel('Position ($\mu$m)')
    fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.046, pad=0.04, label='mV')
    fig.suptitle('$V$', fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()