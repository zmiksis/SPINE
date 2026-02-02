# calcium_wave_new.py
# Example using NEW synapse architecture
# Demonstrates calcium wave propagation with modern SynapseFactory API

from spine.utils import NeuronModel
from spine.solver import SBDF
from spine.graph import gen_beam
from spine.synapse import SynapseFactory
from spine.utils.threading import limit_to_one_thread

import matplotlib.pyplot as plt
import numpy as np
import time

def main():

    # Create neuron model instance
    neuron = NeuronModel()

    # Top data directory, where problem setup and results are stored
    neuron.settings.top_folder = './'

    # Subdirectory with data for this specific problem
    neuron.settings.data_folder = 'calcium_wave/'
    neuron.settings.output_folder = 'calcium_wave-new-output/'

    # Settings for coupled calcium model
    neuron.model['CYT_buffer'] = True
    neuron.model['PMCA'] = True
    neuron.model['NCX'] = True
    neuron.model['SERCA'] = True
    neuron.model['ER_leak'] = True
    neuron.model['RyR'] = True
    neuron.model['IP3'] = True
    neuron.model['synapse'] = True

    # Record all cytosolic calcium data
    neuron.recorder['full'] = True
    neuron.recorder['full_er'] = True

    # Create beam neuron
    neuron.geom_builder['const_rad'] = True
    neuron.settings.rad = 0.2  # um
    gen_beam(neuron, 64, 1000)  # um length, 1000 nodes

    # Final time (s)
    neuron.settings.T = 70.e-3  # seconds
    # Fixed time step
    neuron.settings.set_time_step(5.e-6)  # seconds

    # Create calcium and IP3 synapses using NEW SynapseFactory
    # Linear decay calcium influx at node 500
    syn_ca = SynapseFactory.create_calcium_linear(
        nodes=[500],
        amplitude=2.5e-12,
        duration=1.e-3
    )

    # Linear decay IP3 influx at node 500
    syn_ip3 = SynapseFactory.create_ip3_linear(
        nodes=[500],
        amplitude=5.0e-12,
        duration=200.e-3
    )

    neuron.synapse_instances = [syn_ca, syn_ip3]

    # Run solver
    time0 = time.time()
    SBDF(neuron)
    time1 = time.time()

    print(f"Time taken for SBDF: {time1 - time0} seconds")

    # Load and plot results
    C = np.loadtxt('results/'+neuron.settings.output_folder+'cyt.txt')
    CE = np.loadtxt('results/'+neuron.settings.output_folder+'er.txt')
    x = np.linspace(0, 64, C.shape[1])
    t = np.linspace(0, 70, C.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Cytosolic calcium
    im1 = axes[0].imshow(np.flipud(C.T),
                        extent=[t[0], t[-1], x[0], x[-1]],
                        aspect='auto',
                        cmap='jet', vmin=0.0, vmax=7.0)
    axes[0].set_ylabel('Position ($\mu$m)')
    axes[0].set_title('$Ca^{2+}_{cyt}$')
    fig.colorbar(im1, ax=axes[0], label='$\mu$M')

    # ER calcium
    im2 = axes[1].imshow(np.flipud(CE.T),
                        extent=[t[0], t[-1], x[0], x[-1]],
                        aspect='auto',
                        cmap='jet', vmin=0.0, vmax=275.0)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Position ($\mu$m)')
    axes[1].set_title('$Ca^{2+}_{er}$')
    fig.colorbar(im2, ax=axes[1], label='$\mu$M')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    limit_to_one_thread()
    main()
