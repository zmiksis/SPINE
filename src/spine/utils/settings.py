from dataclasses import dataclass
import numpy as np
import os

__all__ = ['ProblemSettings', 'SolutionStruct', 'model', 'recorder', 'geom_builder']

# Model settings
model = {
	'CYT_buffer' : False,
	'PMCA' : False,
	'NCX' : False,
	'VDCC' : False,
	'VDCC_type' : 'N', # Options are: N, T, L
	'synapse' : False, # This is prototype synapse, not real model (e.g. no AMPA, NMDA, GABA, etc.)
	'PM_leak' : False,
	'ER_leak' : False,
	'ER_buffer' : False,
	'RyR' : False,
	'SOC' : False,
	'SERCA' : False,
	'IP3' : False, # Not implemented for FJ model
	'voltage_coupling' : False,
}

###########################################################

recorder = {
	# Concentration recordings
	'soma' : False, # Write soma concentration to file
	'avg_cyt' : False, # Write average concentration to file
	'avg_er' : False, # Write average ER concentration to file
	'full' : False, # Write full cytosolic concentration to file
	'full_er' : False, # Write full ER concentration to file
	'full_voltage' : False, # Write full voltage data to file
	'nodes' : [], # List of node indices to record concentration data
	'flux_nodes' : [], # List of node indices to record flux data (more memory efficient than full flux arrays)
	'total_cyt' : False, # Write total cytosolic calcium to file
	'total_er' : False, # Write total ER calcium to file

	# Individual flux recordings (memory-efficient system)
	# NOTE: By default, flux flags write ALL nodes to separate files (e.g., flux_pmca.txt)
	#       To write only SPECIFIC nodes, use 'flux_nodes' instead (more efficient)
	#       When 'flux_nodes' is set, fluxes are written to flux_node_{n}.txt files
	#       Example: flux_nodes=[100, 200, 500] + flux_pmca=True + flux_ryr=True
	#                writes pmca and ryr fluxes for nodes 100, 200, 500 only
	#
	# Cytosolic/PM fluxes
	'flux_pmca' : False, # Record PMCA pump flux
	'flux_ncx' : False, # Record NCX exchanger flux
	'flux_pm_leak' : False, # Record PM leak flux
	'flux_vdcc' : False, # Record VDCC flux
	'flux_synapse' : False, # Record synaptic calcium flux
	'flux_synapse_ip3' : False, # Record synaptic IP3 flux

	# ER fluxes
	'flux_ryr' : False, # Record RyR flux (replaces old 'RyR' key)
	'flux_serca' : False, # Record SERCA pump flux
	'flux_er_leak' : False, # Record ER leak flux
	'flux_ip3r' : False, # Record IP3 receptor flux
	'flux_soc' : False, # Record SOC flux

	# Aggregate fluxes
	'flux_total_pm' : False, # Record total PM flux (JPM)
	'flux_total_er' : False, # Record total ER flux (JER)
	'flux_total_ip3' : False, # Record total IP3 flux (JIP3)

	# Legacy (deprecated, use flux_ryr instead)
	'RyR' : [], # DEPRECATED: List of node indices to record RyR state (use flux_ryr instead)

	# VTU visualization output fields (ParaView/VTK format)
	# Set to False to exclude from VTU files (reduces file size and I/O time)
	'vtu_cyt' : True, # Cytosolic calcium concentration
	'vtu_er' : True, # ER calcium concentration
	'vtu_calb' : True, # Calcium buffer concentration
	'vtu_ip3' : True, # IP3 concentration (auto-disabled if IP3 model off)
	'vtu_volt' : True, # Membrane voltage
	'vtu_radius' : True, # Dendrite radius
	'vtu_total_cyt' : True, # Integrated cytosolic calcium
	'vtu_total_er' : True, # Integrated ER calcium
}

###########################################################

geom_builder = {
    'const_rad' : False, # Constant dendritic radius
	'const_syn' : False, # Constant synaptic current
	'const_edge' : False, # Constant edge length
}

###########################################################

# Class for storing solution data
@dataclass
class SolutionStruct:
	C: float = None
	B: float = None
	CE: float = None
	BE: float = None
	IP3: float = None
	V: float = None
	soma: float = None
	avg: float = None
	avg_er: float = None
	total_cyt: float = None
	total_er: float = None

###########################################################

class ProblemSettings():
	def __init__(self, read_voltage=None):
		# Time settings
		self.T0 = 0.
		self.T = None
		self.time_steps = 0
		self.dt = 0.
		self.ntime = 0

		# Diffusion coefficients
		self.Dc = 220. # um^2/s
		self.Db = 20. # um^2/s
		self.De = 220. # um^2/s
		self.Dbe = 27. # um^2/s
		self.Dp = 280. # um^2/s

		# Equilibrium concentrations
		self.cceq = 50.e-18 # umol / um^3
		self.ceeq = 250.e-15 # umol / um^3
		self.pr = 40.e-18 # umol / um^3

		# Top data directory, where problem setup and results are stored
		# Automatically set to current directory if not specified
		self.top_folder = './'
		# Subdirectory with data for this specific problem
		self.data_folder = None
		# Folder where results are stored under top_folder/results/
		self.output_folder = None

		# Voltage data directory
		self.Vdat = None
		self.volt_scale = 1.0e-3 # mV to V
		self.voltage_loop = None
		self.nvolt = 0
		self._read_voltage = read_voltage or self.default_read_voltage
		self.volt_vector = None

		# Voltage problem settings
		self.Vrest = -72.e-3 # V
		self.Cm = 0.01e-12 # F / um^2
		self.Rm = 750.0e4 # Ohm x um

		# Set location of a voltage clamp
		self.voltage_clamp = None # (node_index, voltage_value)

		# Geometry settings
		self.len_scale = 1.0e6 # m to um
		self.rad = 0.5 # if constant radius, um
		self.h = 1.0 # if constant edge length, um

	def set_time_step(self, dt):
		self.dt = dt
		self.ntime = int(np.ceil(self.T/self.dt))

	def set_voltage_loop(self):
		if self.voltage_loop is None and self.Vdat is not None: 
			self.nvolt = sum(1 for f in os.listdir(self.Vdat) if os.path.isfile(os.path.join(self.Vdat, f))) - 1
		elif self.voltage_loop is None and self.Vdat is None:
			self.nvolt = 0
		else: 
			self.nvolt = int(self.voltage_loop/self.dt)

	def read_voltage(self, *args, **kwargs):
		return self._read_voltage(self, *args, **kwargs)

	def default_read_voltage(self, *args, **kwargs):
		tid = kwargs.get('tid', None)
		length = kwargs.get('length', None)
		if self.Vdat is None: V0 = -72.0e-3*np.ones(length)
		else: V0 = self.volt_scale*np.loadtxt(self.Vdat+'vm_'+f"{tid%self.nvolt:07d}"+'.dat', usecols=3)
		return V0
