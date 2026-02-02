# Import module for generating neuron model
from spine.utils import NeuronModel
# Import solver 
from spine.solver import SBDF
# Import graph generation functions
from spine.graph import gen_beam
# Import synapse class
from spine.synapse import synapse
# Import threading utility
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
    neuron.settings.output_folder = 'calcium_wave-output/'

    # Settings for coupled calcium model
    neuron.model['CYT_buffer'] = True
    neuron.model['PMCA'] = True
    neuron.model['NCX'] = True
    neuron.model['SERCA'] = True
    neuron.model['ER_leak'] = True
    neuron.model['RyR'] = True
    neuron.model['IP3'] = True
    neuron.model['synapse'] = True

    # Record all cytosolic calsium data
    neuron.recorder['full'] = True
    neuron.recorder['full_er'] = True

    # Create beam neuron
    neuron.geom_builder['const_rad'] = True
    neuron.settings.rad = 0.2 # um
    gen_beam(neuron, 64, 1000) # um length, 1000 nodes

    # Final time (s)
    neuron.settings.T = 70.e-3 # seconds
    # Fixed time step
    neuron.settings.set_time_step(5.e-6) # seconds

    # Calcium influx at node 0
    syn = synapse()
    syn.type = 'linear'
    syn.duration = 1.e-3 # seconds
    syn.j = 2.5e-12
    syn.node = [500]
    syn.domain = 'cytosol'
    neuron.synapse_instances = [syn]

    # IP3 influx at node 0
    syn_ip3 = synapse()
    syn_ip3.type = 'linear'
    syn_ip3.duration = 200.e-3 # seconds
    syn_ip3.j = 5.0e-12
    syn_ip3.node = [500]
    syn_ip3.domain = 'ip3'
    neuron.synapse_instances.append(syn_ip3)

    # Run solver
    time0 = time.time()
    SBDF(neuron)
    time1 = time.time()

    print(f"Time taken for SBDF: {time1 - time0} seconds")

    # Load and plot results
    C = np.loadtxt('results/'+neuron.settings.output_folder+'cyt.txt')
    CE = np.loadtxt('results/'+neuron.settings.output_folder+'er.txt')
    x = np.linspace(0, 64, C.shape[1])

    step = 1
    legend_vec = []
    for i in range(1000,10001,1000):
        plt.plot(x, C[i,:])
        legend_vec.append(str(step*5)+' ms')
        step += 1

    plt.xlabel('Length ($\mu$m)')
    plt.ylabel('Ca$^{2+}$ ($\mu$M)')
    plt.ylim((0,np.max(np.max(C))+2))
    plt.legend(legend_vec, loc='upper left', ncol=4)
    plt.show()

    C = C[:6001,:]  # Plot only first 30 ms
    fig, ax = plt.subplots()
    x = np.linspace(0, 65, C.shape[1])
    t = np.linspace(0, 30, C.shape[0])

    # Create the heatmap
    im = ax.imshow(np.flipud(C.T), 
                    extent=[t[0], t[-1], x[0], x[-1]], 
                    aspect='auto',
                    cmap='jet',
                    vmin=0.0,  
                    vmax=7.0
                    )

    # Add a colorbar
    fig.colorbar(im, 
                 ax=ax, 
                 label='Value')

    # Set labels and title
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Position ($\mu$m)')
    ax.set_title('$Ca^{2+}_{cyt}$ ($\mu$M)')

    plt.show()

    CE = CE[:6001,:]  # Plot only first 30 ms
    fig, ax = plt.subplots()

    # Create the heatmap
    im = ax.imshow(np.flipud(CE.T), 
                    extent=[t[0], t[-1], x[0], x[-1]], 
                    aspect='auto',
                    cmap='jet',
                    vmin=0.0,  
                    vmax=275.0
                    )

    # Add a colorbar
    fig.colorbar(im, 
                 ax=ax, 
                 label='Value')

    # Set labels and title
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Position ($\mu$m)')
    ax.set_title('$Ca^{2+}_{er}$ ($\mu$M)')

    plt.show()

if __name__ == "__main__":
    # Limit to one thread per process
    limit_to_one_thread()
    main()