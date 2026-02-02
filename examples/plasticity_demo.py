# plasticity_demo.py
# Demonstrates calcium-dependent synaptic plasticity using NEW synapse architecture
# Shows paired-pulse facilitation with CalciumModulatedAMPA receptor

from spine.utils import NeuronModel
from spine.solver import SBDF
from spine.graph import gen_beam
from spine.synapse import SynapseFactory, SynapseCommunicator

import matplotlib.pyplot as plt
import numpy as np
import copy

def main():

    print("=" * 60)
    print("Calcium-Dependent Synaptic Plasticity Demonstration")
    print("=" * 60)

    # Create presynaptic neuron
    pre_neuron = NeuronModel()
    pre_neuron.settings.top_folder = './'
    pre_neuron.settings.data_folder = 'plasticity_demo/'
    pre_neuron.settings.output_folder = 'plasticity-pre_output/'

    # Enable calcium dynamics and voltage coupling
    pre_neuron.model['CYT_buffer'] = True
    pre_neuron.model['PMCA'] = True
    pre_neuron.model['NCX'] = True
    pre_neuron.model['SERCA'] = True
    pre_neuron.model['ER_leak'] = True
    pre_neuron.model['RyR'] = True
    pre_neuron.model['IP3'] = True
    pre_neuron.model['voltage_coupling'] = True
    pre_neuron.currents['I_Na'] = True
    pre_neuron.currents['I_K'] = True
    pre_neuron.currents['I_Leak'] = True

    # Recording
    pre_neuron.recorder['full'] = True
    pre_neuron.recorder['full_voltage'] = True
    pre_neuron.recorder['nodes'] = [0, 50, 100]  # Record specific nodes

    # Create beam
    pre_neuron.geom_builder['const_rad'] = True
    pre_neuron.settings.rad = 0.5  # um
    gen_beam(pre_neuron, 100, 200)  # 100 um length, 200 nodes

    # Simulation time
    pre_neuron.settings.T = 100.e-3  # 100 ms
    pre_neuron.settings.set_time_step(10.e-6)  # 10 us

    # Create postsynaptic neuron
    post_neuron = copy.deepcopy(pre_neuron)
    post_neuron.settings.output_folder = 'plasticity-post_output/'

    # ========== SYNAPTIC STIMULATION ==========

    # Paired calcium pulses in presynaptic neuron (paired-pulse protocol)
    # First pulse at 20 ms
    pulse1 = SynapseFactory.create_calcium_pulse(
        nodes=[0],  # First node
        amplitude=5e-12,
        duration=2e-3,  # 2 ms pulse
        start_time=20e-3
    )

    # Second pulse at 50 ms (30 ms ISI)
    pulse2 = SynapseFactory.create_calcium_pulse(
        nodes=[0],
        amplitude=5e-12,
        duration=2e-3,
        start_time=50e-3
    )

    pre_neuron.synapse_instances = [pulse1, pulse2]

    # ========== CALCIUM-MODULATED AMPA SYNAPSE ==========

    # Postsynaptic neuron receives calcium-modulated AMPA input
    # This synapse will show facilitation due to presynaptic calcium buildup
    ampa_facilitation = SynapseFactory.create_calcium_modulated_AMPA(
        post_nodes=[100],  # Middle of postsynaptic neuron
        g_max=3e-10,
        ca_sensitivity=10.0,    # High sensitivity (strong facilitation)
        baseline_release=0.2,   # Low baseline (large dynamic range)
        ca_baseline=0.05       # Resting calcium (μM)
    )

    post_neuron.synapse_instances = [ampa_facilitation]

    # ========== INTER-NEURON COMMUNICATION ==========

    neuronList = [pre_neuron, post_neuron]
    comm = SynapseCommunicator(neuronList)

    # Connect presynaptic node 0 to postsynaptic node 100
    comm.add_synapse_connection((0, 0), (1, 100), weight=1200)

    # Run simulation
    print("\nRunning simulation...")
    SBDF(neuronList, comm=comm, n_workers=2)
    print("Simulation complete!")

    # ========== ANALYSIS ==========

    # Load presynaptic calcium at stimulation site
    pre_data = np.loadtxt('results/' + pre_neuron.settings.output_folder + 'node_0.txt')
    t_pre = pre_data[:, 0] * 1e3  # Convert to ms
    Ca_pre = pre_data[:, 1]  # Cytosolic calcium (μM)

    # Load postsynaptic voltage at synapse location
    post_data = np.loadtxt('results/' + post_neuron.settings.output_folder + 'node_100.txt')
    t_post = post_data[:, 0] * 1e3
    V_post = post_data[:, 4]  # Voltage (mV)

    # ========== PLOTTING ==========

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: Presynaptic calcium
    axes[0].plot(t_pre, Ca_pre, 'b-', linewidth=2, label='Presynaptic Ca²⁺')
    axes[0].axvline(20, color='r', linestyle='--', alpha=0.5, label='Pulse 1')
    axes[0].axvline(50, color='r', linestyle='--', alpha=0.5, label='Pulse 2')
    axes[0].set_ylabel('Presynaptic Ca²⁺ (μM)', fontsize=12)
    axes[0].set_title('Calcium-Dependent Synaptic Facilitation', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: Postsynaptic voltage (shows facilitated response)
    axes[1].plot(t_post, V_post, 'g-', linewidth=2, label='Postsynaptic Voltage')
    axes[1].axvline(20, color='r', linestyle='--', alpha=0.5)
    axes[1].axvline(50, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time (ms)', fontsize=12)
    axes[1].set_ylabel('Postsynaptic V (mV)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    print("\n" + "=" * 60)
    print("Key Observations:")
    print("=" * 60)
    print("1. First pulse (20 ms): Elevates presynaptic calcium")
    print("2. Second pulse (50 ms): Calcium still elevated from first pulse")
    print("3. Result: Second postsynaptic response is FACILITATED")
    print("   (larger amplitude due to calcium-dependent release)")
    print("=" * 60)

    plt.show()

if __name__ == "__main__":
    main()
