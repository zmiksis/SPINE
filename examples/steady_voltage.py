# Import module for generating neuron model
from spine.utils import NeuronModel
# Import solver 
from spine.solver import SBDF
# Import graph generation functions
from spine.graph import gen_beam

import numpy as np
import matplotlib.pyplot as plt

def main():

    # Create neuron model instance
    neuron = NeuronModel()

    # Top data directory, where problem setup and results are stored
    neuron.settings.top_folder = './'

    # Subdirectory with data for this specific problem
    neuron.settings.data_folder = 'steady_voltage/'
    neuron.settings.output_folder = 'steady_voltage-output/'
    neuron.recorder['full_voltage'] = True
    neuron.recorder['full'] = True

    # Enable voltage coupling and ionic currents
    neuron.model['voltage_coupling'] = True
    neuron.currents['I_Na'] = True
    neuron.currents['I_K'] = True
    neuron.currents['I_leak'] = True
    neuron.currents['I_Ca'] = True

    # Enable calcium dynamics for observation if calcium currents are enabled
    if neuron.currents['I_Ca']:
        neuron.model['CYT_buffer'] = True
        neuron.model['PMCA'] = True
        neuron.model['NCX'] = True
        neuron.model['VDCC'] = True
        neuron.model['PM_leak'] = True
        neuron.model['ER_leak'] = True
        neuron.model['SERCA'] = True
        neuron.model['IP3R'] = True
        neuron.model['RyR'] = True

    # Create beam neuron
    neuron.geom_builder['const_rad'] = True
    neuron.settings.rad = 0.2 # um
    gen_beam(neuron, 64, 250) # um length, 250 nodes

    # Final time (s)
    neuron.settings.T = 1. # seconds
    # Fixed time step
    neuron.settings.set_time_step(20.e-6) # seconds

    SBDF(neuron)

    V = np.loadtxt('results/'+neuron.settings.output_folder+'volt.txt')
    t = np.linspace(0, neuron.settings.T, len(V[:,0]))
    plt.plot(t, V[:,0])
    plt.show()

    if neuron.currents['I_Ca']:
        C = np.loadtxt('results/'+neuron.settings.output_folder+'cyt.txt')
        plt.plot(t, C[:,0])
        plt.show()

if __name__== "__main__":
    main()