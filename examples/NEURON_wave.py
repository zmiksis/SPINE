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

def synaptic_release_pattern(self, *args, **kwargs):
    self.t = kwargs.get('t', None)

def initial_conditions(self, *args, **kwargs):
    # This is an output array in uM
    self.sol.IP3[498:503] = 1.25

def main():

    neuron = NeuronModel(initial_conditions=initial_conditions)

    # Top data directory, where problem setup and results are stored
    neuron.settings.top_folder = './'

    # Subdirectory with data for this specific problem
    neuron.settings.data_folder = 'NEURON_wave/'
    neuron.settings.output_folder = 'NEURON_wave-output/'

    # Settings for NEURON-like calcium wave
    neuron.model['IP3'] = True
    neuron.model['ER_leak'] = True
    neuron.model['SERCA'] = True

    # Add voltage coupling
    neuron.model['voltage_coupling'] = True
    neuron.currents['I_Na'] = True
    neuron.currents['I_K'] = True
    neuron.currents['I_Leak'] = True

    # Record all cytosolic calcium data
    neuron.recorder['full'] = True

    # Create beam neuron
    neuron.geom_builder['const_rad'] = True
    neuron.settings.rad = 0.5 # um
    gen_beam(neuron, 1000., 1000) # um length, 1000 nodes

    # Final time (s)
    neuron.settings.T = 30. # seconds
    # Fixed time step
    neuron.settings.set_time_step(5.e-3) # seconds

    neuron.settings.Dc = 0.08e3 # um^2/s
    neuron.settings.De = 0.08e3 # um^2/s
    neuron.settings.Dp = 0.2e3 # um^2/s

    neuron.settings.cceq = 0.0001e-12 # umol / um^3
    neuron.settings.pr = 0.1e-12 # umol / um^3
    cAVG = 0.0017e-12 # umol / um^3
    neuron.settings.ceeq = (cAVG - 0.83 * neuron.settings.cceq) / 0.17 # umol / um^3

    # Run solver
    time0 = time.time()
    SBDF(neuron, writeStep=5, er_scale=0.17/0.83)
    time1 = time.time()

    print(f"Simulation time: {time1 - time0:.2f} seconds")

    data = np.loadtxt('results/' + neuron.settings.output_folder + 'cyt.txt')

    fig, ax = plt.subplots()
    x = np.linspace(0, 1000, data.shape[0])   # e.g., spatial coordinate
    t = np.linspace(0, 30, data.shape[1])    # e.g., time coordinate

    # Create the heatmap
    im = ax.imshow(np.flipud(data.T), 
                    extent=[t[0], t[-1], x[0], x[-1]], 
                    aspect='auto',
                    cmap='viridis',
                    vmin=0.0,  
                    vmax=0.1
                    )

    # Add a colorbar
    fig.colorbar(im, 
                 ax=ax, 
                 label='Value')

    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position ($\mu$m)')
    ax.set_title('$Ca^{2+}$ ($\mu$M)')

    plt.show()

if __name__ == "__main__":
    limit_to_one_thread()
    main()