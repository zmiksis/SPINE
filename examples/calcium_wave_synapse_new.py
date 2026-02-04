# calcium_wave_synapse_new.py
# Example using NEW synapse architecture
# Demonstrates inter-neuron calcium coupling with calcium-dependent synaptic transmission

from spine.utils import NeuronModel
from spine.solver import SBDF
from spine.graph import gen_beam
from spine.synapse import SynapseFactory, SynapseCommunicator
from spine.utils.settings import ProblemSettings

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--proc", type=int)
    args = parser.parse_args()

    # Setting number of processes
    if args.proc is not None:
        proc = args.proc
    else:
        proc = 1

    # Create first neuron (presynaptic)
    neuron1 = NeuronModel()

    # Customize voltage reading function
    neuron1.settings = ProblemSettings(read_voltage=custom_voltage_reader)
    neuron1.settings.volt_vector = np.loadtxt('volt.txt')

    # Top data directory
    neuron1.settings.top_folder = './'
    neuron1.settings.data_folder = 'calcium_wave_synapse/'
    neuron1.settings.output_folder = 'calcium_wave_synapse-new-pre_output/'

    # Settings for coupled calcium model
    neuron1.model['CYT_buffer'] = True
    neuron1.model['PMCA'] = True
    neuron1.model['NCX'] = True
    neuron1.model['PM_leak'] = True
    neuron1.model['SERCA'] = True
    neuron1.model['ER_leak'] = True
    neuron1.model['RyR'] = True
    neuron1.model['IP3'] = True
    neuron1.model['synapse'] = True

    # Record all data
    neuron1.recorder['full'] = True
    neuron1.recorder['full_er'] = True
    neuron1.recorder['full_voltage'] = True

    # Create beam neuron
    neuron1.geom_builder['const_rad'] = True
    neuron1.settings.rad = 0.2  # um
    gen_beam(neuron1, 64, 1001)  # um length, 1001 nodes

    # Final time (s)
    neuron1.settings.T = 60.e-3
    neuron1.settings.set_time_step(5.e-6)

    # Create second neuron (postsynaptic) as a copy
    neuron2 = copy.deepcopy(neuron1)
    neuron2.settings.output_folder = 'calcium_wave_synapse-new-post_output/'

    # Turn on voltage coupling and ionic currents for postsynaptic neuron
    neuron2.model['voltage_coupling'] = True
    neuron2.currents['I_Na'] = True
    neuron2.currents['I_K'] = True
    neuron2.currents['I_Leak'] = True
    neuron2.currents['I_Ca'] = True
    neuron2.currents['I_syn'] = True

    # ========== NEW SYNAPSE ARCHITECTURE ==========

    # Presynaptic neuron: Calcium and IP3 injection at midpoint
    syn_ca = SynapseFactory.create_calcium_linear(
        nodes=[500],
        amplitude=2.5e-12,
        duration=1.e-3
    )

    syn_ip3 = SynapseFactory.create_ip3_linear(
        nodes=[500],
        amplitude=5.0e-12,
        duration=200.e-3
    )

    syn_ampa = SynapseFactory.create_AMPA_voltage_modulated(
        nodes=[500]
    )

    # neuron1.synapse_instances = [syn_ca, syn_ip3]
    neuron1.synapse_instances = [syn_ampa]

    # Postsynaptic neuron: Calcium-dependent coupling from presynaptic neuron
    # This synapse responds to presynaptic calcium at node 0
    # Note: CalciumCoupledSynapse is always active (no duration parameter)
    syn_coupling = SynapseFactory.create_calcium_coupled(
        post_nodes=[1000],  # end node of postsynaptic neuron
        sensitivity=1e3     # sensitivity to presynaptic calcium (s^-1)
    )

    neuron2.synapse_instances = [syn_coupling]

    # ========== INTER-NEURON COMMUNICATION ==========

    neuronList = [neuron1, neuron2]
    communicator = SynapseCommunicator(neuronList)

    # Synapse connection parameters (from original)
    receptors_per_synapse = 20
    synapses_per_cluster = 20
    neurons_per_cluster = 1
    density = 3
    total_receptors = receptors_per_synapse * synapses_per_cluster * neurons_per_cluster * density

    # Connect presynaptic neuron node 0 to postsynaptic neuron node 1000
    communicator.add_synapse_connection((0, 0), (1, 1000), total_receptors)

    # Run solver
    time0 = time.time()
    SBDF(neuronList, comm=communicator, n_workers=proc, threads_per_process=1)
    time1 = time.time()

    print(f"Time taken for solver: {time1 - time0} seconds")

    # Load results
    C1 = np.loadtxt('results/' + neuron1.settings.output_folder + 'cyt.txt')
    CE1 = np.loadtxt('results/' + neuron1.settings.output_folder + 'er.txt')
    V1 = np.loadtxt('results/' + neuron1.settings.output_folder + 'volt.txt')

    C2 = np.loadtxt('results/' + neuron2.settings.output_folder + 'cyt.txt')
    CE2 = np.loadtxt('results/' + neuron2.settings.output_folder + 'er.txt')
    V2 = np.loadtxt('results/' + neuron2.settings.output_folder + 'volt.txt')

    x = np.linspace(0, 64, C1.shape[1])
    t = np.linspace(0, 60, C1.shape[0])

    # Plot cytosolic calcium
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

    # Plot ER calcium
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

    # Plot voltage
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

if __name__ == "__main__":
    main()
