from spine.utils import NeuronModel
from spine.solver import SBDF
from spine.graph import gen_beam
from spine.synapse import synapse
from spine.synapse import SynapseCommunicator
from spine.utils.settings import ProblemSettings

import matplotlib.pyplot as plt
import numpy as np
import copy

def custom_voltage_reader(self, *args, **kwargs):
    tid = kwargs.get('tid', None)
    length = kwargs.get('length', None)
    V0 = self.volt_scale*self.volt_vector[tid,1]*np.ones(length)
    return V0

def release_profile(self, *args, **kwargs):
    comm = kwargs.get('comm', None)
    comm_iter = kwargs.get('comm_iter', None)
    node = kwargs.get('node', None)
    neuron = kwargs.get('neuron', None)
    C, comm_node = comm.get_pre_C0(comm_iter)
    if node not in comm_node:
        return 0.0
    return 1e3 * (1.e-15*C[comm_node.index(node)] - neuron.settings.cceq)

def main():

    neuron = NeuronModel()

    neuron.settings = ProblemSettings(read_voltage=custom_voltage_reader)
    neuron.settings.volt_vector = np.loadtxt('volt.txt')

    neuron.settings.top_folder = './'
    neuron.settings.data_folder = 'calcium_wave_synapse/'
    neuron.settings.output_folder = 'calcium_wave_synapse-output/'

    neuron.model['CYT_buffer'] = True
    neuron.model['PM_leak'] = True
    neuron.model['PMCA'] = True
    neuron.model['NCX'] = True
    neuron.model['SERCA'] = True
    neuron.model['ER_leak'] = True
    neuron.model['RyR'] = True
    neuron.model['IP3'] = True
    neuron.model['synapse'] = True

    neuron.recorder['full'] = True
    neuron.recorder['full_er'] = True
    neuron.recorder['full_voltage'] = True

    neuron.geom_builder['const_rad'] = True
    neuron.settings.rad = 0.2
    gen_beam(neuron, 64, 1001)

    neuron.settings.T = 60.e-3 
    neuron.settings.set_time_step(5.e-6) 

    neuron_copy = copy.deepcopy(neuron)
    neuron.settings.data_folder = 'calcium_wave_synapse/'
    neuron_copy.settings.output_folder = 'calcium_wave_synapse-copy_output/'

    neuron_copy.model['voltage_coupling'] = True
    neuron_copy.currents['I_Na'] = True
    neuron_copy.currents['I_K'] = True
    neuron_copy.currents['I_Leak'] = True
    neuron_copy.currents['I_Ca'] = True
    neuron_copy.currents['I_syn'] = True

    syn = synapse()
    syn.type = 'linear'
    syn.duration = 1.e-3
    syn.j = 2.5e-12
    syn.node = [500]
    syn.domain = 'cytosol'

    syn_ip3 = synapse()
    syn_ip3.type = 'linear'
    syn_ip3.duration = 200.e-3
    syn_ip3.j = 5.0e-12
    syn_ip3.node = [500]
    syn_ip3.domain = 'ip3'

    neuron.synapse_instances = [syn, syn_ip3]

    syn_comm = synapse(release_profile=release_profile)
    syn_comm.domain = 'receptor'
    syn_comm.duration = neuron.settings.T
    syn_comm.node = [1000] 
    neuron_copy.synapse_instances = [syn_comm]

    neuronList = [neuron, neuron_copy]
    communicator = SynapseCommunicator(neuronList)

    receptors_per_synapse = 20
    synapses_per_cluster = 20
    neurons_per_cluster = 1
    density = 3
    total_receptors = receptors_per_synapse \
                    * synapses_per_cluster \
                    * neurons_per_cluster \
                    * density

    communicator.add_synapse_connection((0, 0), (1, 1000), total_receptors)

    SBDF(neuronList, comm=communicator, n_workers=2, threads_per_process=1)

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