# Synapse communicator to handle data exchange between neurons
from multiprocessing import shared_memory
import numpy as np

class SynapseCommunicator:
    def __init__(self, neuron_list):
        self.neuron_list = neuron_list
        self.pre_post_map = {}
        self.pre_post_node_map = {}
        self.weight_map = {}
        self._attach_shared_memory()

    def _attach_shared_memory(self):
        """Attach to all neurons' shared memory once during initialization."""
        for neuron in self.neuron_list:
            # Attach to C0 shared memory
            if hasattr(neuron, 'C0_name'):
                neuron.shmC0 = shared_memory.SharedMemory(name=neuron.C0_name)
                neuron.sol.C0 = np.ndarray(neuron.C0_shape, dtype=neuron.C0_dtype, buffer=neuron.shmC0.buf)

            # Attach to V0 shared memory
            if hasattr(neuron, 'V0_name'):
                neuron.shmV0 = shared_memory.SharedMemory(name=neuron.V0_name)
                neuron.sol.V0 = np.ndarray(neuron.V0_shape, dtype=neuron.V0_dtype, buffer=neuron.shmV0.buf)

            # Attach to V shared memory
            if hasattr(neuron, 'V_name'):
                neuron.shmV = shared_memory.SharedMemory(name=neuron.V_name)
                neuron.sol.V = np.ndarray(neuron.V_shape, dtype=neuron.V_dtype, buffer=neuron.shmV.buf)

    def add_synapse_connection(self, pre_tuple, post_tuple, weight=1.0):

        pre_neuron, pre_node_indices = pre_tuple
        post_neuron, post_node_indices = post_tuple
        
        if f'{post_neuron}' not in self.pre_post_map:
            self.pre_post_map[f'{post_neuron}'] = []
            self.pre_post_node_map[f'{post_neuron}'] = {}
            self.weight_map[f'{post_neuron}'] = {}

        if f'{pre_neuron}' not in self.pre_post_node_map[f'{post_neuron}']:
            self.pre_post_node_map[f'{post_neuron}'][f'{pre_neuron}'] = []
            self.weight_map[f'{post_neuron}'][f'{pre_neuron}'] = []

        if pre_neuron not in self.pre_post_map[f'{post_neuron}']:
            self.pre_post_map[f'{post_neuron}'].append(pre_neuron)

        self.pre_post_node_map[f'{post_neuron}'][f'{pre_neuron}'].append((pre_node_indices, post_node_indices))
        self.weight_map[f'{post_neuron}'][f'{pre_neuron}'].append(weight)

    def get_pre_C0(self, post_neuron):
        """Get presynaptic calcium data for postsynaptic neuron."""
        pre_neuron_data_list = []
        post_neuron_node_list = []
        for pre_neuron in self.pre_post_map[f'{post_neuron}']:
            for (pre_nodes, post_nodes) in self.pre_post_node_map[f'{post_neuron}'][f'{pre_neuron}']:
                pre_neuron_data_list.append(self.neuron_list[pre_neuron].sol.C0[pre_nodes])
                post_neuron_node_list.append(post_nodes)
        return pre_neuron_data_list, post_neuron_node_list
    
    def get_pre_V0(self, post_neuron):
        """Get presynaptic V0 data for postsynaptic neuron."""
        pre_neuron_data_list = []
        post_neuron_node_list = []
        for pre_neuron in self.pre_post_map[f'{post_neuron}']:
            for (pre_nodes, post_nodes) in self.pre_post_node_map[f'{post_neuron}'][f'{pre_neuron}']:
                pre_neuron_data_list.append(self.neuron_list[pre_neuron].sol.V0[pre_nodes])
                post_neuron_node_list.append(post_nodes)
        return pre_neuron_data_list, post_neuron_node_list
    
    def get_pre_V(self, post_neuron):
        """Get presynaptic voltage data for postsynaptic neuron."""
        pre_neuron_data_list = []
        post_neuron_node_list = []
        for pre_neuron in self.pre_post_map[f'{post_neuron}']:
            for (pre_nodes, post_nodes) in self.pre_post_node_map[f'{post_neuron}'][f'{pre_neuron}']:
                pre_neuron_data_list.append(self.neuron_list[pre_neuron].sol.V[pre_nodes])
                post_neuron_node_list.append(post_nodes)
        return pre_neuron_data_list, post_neuron_node_list