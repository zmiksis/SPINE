from spine.utils.settings import *
from spine.graph import *
from spine import er
from spine import cytosol as cyt
from spine.cable_eq import CableSettings, currents

import os
import numpy as np
from multiprocessing import shared_memory

class NeuronModel():
    def __init__(self, initial_conditions=None):

        self.settings = ProblemSettings()
        self.sol = SolutionStruct()
        self.recorder = recorder.copy()
        self.model = model.copy()

        self.graph = Graph()
        self.geom_builder = geom_builder.copy()

        self.cytExchangeParams = cyt.params.copy()
        self.cytConstants = cyt.constants.copy()

        self.erExchangeParams = er.params.copy()
        self.erConstants = er.constants.copy()

        self.synapse_instances = []

        # if self.model['voltage_coupling']:
        self.currents = currents.copy()
        self.CableSettings = CableSettings(self)

        self._initial_conditions = initial_conditions or self.default_initial_conditions
        
    ###########################################################

    def initialize_shared_memory(self):

        # Create directories for data
        if self.settings.data_folder is not None:
            if not os.path.exists(self.settings.top_folder+self.settings.data_folder): os.mkdir(self.settings.top_folder+self.settings.data_folder)
            if not os.path.exists(self.settings.top_folder+'results'): os.mkdir(self.settings.top_folder+'results')
            if not os.path.exists(self.settings.top_folder+'results/'+self.settings.output_folder): os.mkdir(self.settings.top_folder+'results/'+self.settings.output_folder)
            if not os.path.exists(self.settings.top_folder+'results/'+self.settings.output_folder+'/vtu'): os.mkdir(self.settings.top_folder+'results/'+self.settings.output_folder+'/vtu')

        # Initial condition (store solution in uM)
        self.shmC = shared_memory.SharedMemory(create=True, size=self.graph.nid.size*self.graph.nid.itemsize)
        self.C_name = self.shmC.name
        self.C_shape = self.graph.nid.shape
        self.C_dtype = np.float64
        self.sol.C = np.ndarray(self.C_shape, dtype=self.C_dtype, buffer=self.shmC.buf)
        self.sol.C[:] = 1e15*self.settings.cceq 

        if self.recorder['full'] and self.settings.output_folder is not None:
            np.savetxt(self.settings.top_folder+'results/'+self.settings.output_folder+'/cyt.txt', self.sol.C.reshape(1,-1))

        self.shmC0 = shared_memory.SharedMemory(create=True, size=self.graph.nid.size*self.graph.nid.itemsize)
        self.C0_name = self.shmC0.name
        self.C0_shape = self.graph.nid.shape
        self.C0_dtype = np.float64
        self.sol.C0 = np.ndarray(self.C0_shape, dtype=self.C0_dtype, buffer=self.shmC0.buf)
        self.sol.C0[:] = 1e15*self.settings.cceq

        # Voltage initial condition (store solution in mV)
        self.shmV = shared_memory.SharedMemory(create=True, size=self.graph.nid.size*self.graph.nid.itemsize)
        self.V_name = self.shmV.name
        self.V_shape = self.graph.nid.shape
        self.V_dtype = np.float64
        self.sol.V = np.ndarray(self.V_shape, dtype=self.V_dtype, buffer=self.shmV.buf)
        if self.settings.Vdat is None: self.sol.V[:] = 1e3*self.settings.Vrest
        else: self.sol.V[:] =  1e3*np.loadtxt(self.settings.Vdat+'vm_0000000.dat', usecols=3)
        

        if self.recorder['full_voltage'] and self.settings.output_folder is not None:
            np.savetxt(self.settings.top_folder+'results/'+self.settings.output_folder+'/volt.txt', self.sol.V.reshape(1,-1))

        self.shmV0 = shared_memory.SharedMemory(create=True, size=self.graph.nid.size*self.graph.nid.itemsize)
        self.V0_name = self.shmV0.name
        self.V0_shape = self.graph.nid.shape
        self.V0_dtype = np.float64
        self.sol.V0 = np.ndarray(self.V0_shape, dtype=self.V0_dtype, buffer=self.shmV0.buf)
        if self.settings.Vdat is None: self.sol.V0[:] = 1e3*self.settings.Vrest
        else: self.sol.V0[:] =  1e3*np.loadtxt(self.settings.Vdat+'vm_0000000.dat', usecols=3)

    ###########################################################

    def initialize_problem(self):

        # Time step
        self.settings.set_voltage_loop()

        # Create data arrays
        binit = cyt.init_calb(self, self.settings.cceq)
        self.sol.B = 1e15*binit*np.ones_like(self.graph.nid)
        self.sol.CE = 1e15*self.settings.ceeq*np.ones_like(self.graph.nid)
        beinit = er.init_calbe(self, self.settings.ceeq)
        self.sol.BE = 1e15*beinit*np.ones_like(self.graph.nid)
        self.sol.IP3 = 1e15*self.settings.pr*np.ones_like(self.graph.nid)

        if self.recorder['full_er'] and self.settings.output_folder is not None:
            np.savetxt(self.settings.top_folder+'results/'+self.settings.output_folder+'/er.txt', self.sol.CE.reshape(1,-1))

        if self.recorder['nodes'] and self.settings.output_folder is not None:
            for n in self.recorder['nodes']:
                filename = self.settings.top_folder+'results/'+self.settings.output_folder+f'/node_{n}.txt'
                np.savetxt(filename, np.array([0., self.sol.C[n], self.sol.CE[n], self.sol.IP3[n], self.sol.V[n]]).reshape(1,-1))

        # Initialize node-specific flux recording files
        if self.recorder['flux_nodes'] and self.settings.output_folder is not None:
            for n in self.recorder['flux_nodes']:
                filename = self.settings.top_folder+'results/'+self.settings.output_folder+f'/flux_node_{n}.txt'
                # Initialize with header (time column + enabled flux columns will be added dynamically)
                # First write is at t=0 with zero fluxes
                np.savetxt(filename, np.array([0.]).reshape(1,-1))

        # Legacy RyR recording (deprecated, use flux_ryr instead)
        if self.recorder['RyR'] and self.settings.output_folder is not None:
            np.savetxt(self.settings.top_folder+'results/'+self.settings.output_folder+'/ryr.txt', np.zeros(len(self.recorder['RyR'])).reshape(1,-1))

        # Initialize flux recording files (new system writes headers automatically on first append)
        # No initialization needed - files will be created on first write with 'ab' mode

        if self.recorder['total_cyt'] and self.settings.output_folder is not None:
            self.sol.total_cyt = np.zeros_like(self.sol.C)

        if self.recorder['total_er'] and self.settings.output_folder is not None:
            self.sol.total_er = np.zeros_like(self.sol.CE)
        
    ###########################################################

    def initial_conditions(self, *args, **kwargs):
        return self._initial_conditions(self, *args, **kwargs)
    
    ###########################################################

    def default_initial_conditions(self, *args, **kwargs):
        pass

    ###########################################################

    def swap_history_arrays(self):
        """Swap current and history arrays using pointer swapping.

        This avoids copying data between C/C0 and V/V0. Instead of:
            C0[:] = C[:]  # Copy all data
            C[:] = new_data
        We do:
            swap(C, C0)  # Just swap references
            C[:] = new_data  # C now points to what was C0

        This eliminates 50% of memory writes per timestep.
        """
        # Swap calcium arrays
        self.sol.C, self.sol.C0 = self.sol.C0, self.sol.C
        self.shmC, self.shmC0 = self.shmC0, self.shmC
        self.C_name, self.C0_name = self.C0_name, self.C_name

        # Swap voltage arrays
        self.sol.V, self.sol.V0 = self.sol.V0, self.sol.V
        self.shmV, self.shmV0 = self.shmV0, self.shmV
        self.V_name, self.V0_name = self.V0_name, self.V_name

    ###########################################################

    # Multiprocessing shared memory cleanup
    def close(self):
        self.shmC.close()
        self.shmC0.close()
        self.shmV.close()
        self.shmV0.close()

    def unlink(self):
        try: self.shmC.unlink()
        except FileNotFoundError: pass
        try: self.shmV.unlink()
        except FileNotFoundError: pass
        try: self.shmC0.unlink()
        except FileNotFoundError: pass
        try: self.shmV0.unlink()
        except FileNotFoundError: pass
