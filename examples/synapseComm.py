# Import module for generating neuron model
from spine.utils import NeuronModel
# Import synapse communicator
from spine.synapse import SynapseCommunicator
# Import graph generation functions
from spine.graph import gen_beam, gen_neuron

# Create two beam neuron model instances with different initial calcium concentrations
neuron0 = NeuronModel()
gen_beam(neuron0, 64, 250) # um length, 250 nodes
gen_neuron(neuron0)
neuron0.settings.cceq = 15.e-18
neuron0.initialize_shared_memory()
neuron0.initialize_problem()

neuron1 = NeuronModel()
gen_beam(neuron1, 64, 250) # um length, 250 nodes
gen_neuron(neuron1)
neuron1.settings.cceq = 35.e-18
neuron1.initialize_shared_memory()
neuron1.initialize_problem()

# Create list of neurons
neuronList = [neuron0, neuron1]

# Create synapse communicator
communicator = SynapseCommunicator(neuronList)

# Add synapse connection from neuron 0, node 0 to neuron 1, node 1
communicator.add_synapse_connection((0, 0), (1, 1))
# Add synapse connection from neuron 0, node 1 to neuron 1, node 0
communicator.add_synapse_connection((0, 1), (1, 0))

# Get calcium data for neuron 1 from presynaptic neurons
c, post_nodes = communicator.get_pre_C0(1)
for idx, c_value in zip(post_nodes, c):
    print(f"Post-synaptic node index: {idx}")
    print(f"Presynaptic calcium concentration: {c_value} uM")

neuron0.close()
neuron0.unlink()
neuron1.close()
neuron1.unlink()