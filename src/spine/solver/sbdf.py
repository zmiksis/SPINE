from spine.utils.threading import apply_thread_limits_if_needed
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory, Process, Barrier
import platform

from spine.graph import gen_neuron
from spine.solver.discretization_matrices import *
from spine.solver.helper import *

###########################################################

def SBDF(neuron_list, comm = None, writeStep = None, solver='SIAP', er_scale = 15./40., n_workers=None, threads_per_process=1):

	if not isinstance(neuron_list, list):
		neuron_list = [neuron_list]

	print('======================================================')
	for index, neuron in enumerate(neuron_list):
		print(f"Neuron {index+1} model settings:")
		print(neuron.model)
		print(neuron.cytExchangeParams)
		print(neuron.erExchangeParams)
		print('\n')
	print('Running SBDF2 solver with ' + solver + ' model')
	print('======================================================')

	system = platform.system().lower()
	if system == "linux":
		desired = "fork"
	else:
		desired = "spawn"

	mp.set_start_method(desired, force=True)
	processor = Processor(neuron_list, comm=comm, writeStep=writeStep, solver=solver, er_scale=er_scale, n_workers=n_workers, threads_per_process=threads_per_process)
	processor.shutdown()

###########################################################

class Processor:
	def __init__(self, neuron_list, comm=None, writeStep=100, solver='SIAP', er_scale=15./40., n_workers=None, threads_per_process=1):

		self.neuron_list = neuron_list
		self.comm = comm
		self.writeStep = writeStep
		self.solver = solver
		self.er_scale = er_scale
		self.threads_per_process = threads_per_process

		if n_workers is None:
			self.n_workers = 1
		else:
			self.n_workers = n_workers

		for neuron in self.neuron_list:
			neuron.initialize_shared_memory()

		# Round-robin batching ensures exactly n_workers batches
		self.batches = [[] for _ in range(self.n_workers)]
		self.batch_comm_idx = [[] for _ in range(self.n_workers)]
		for idx, neuron in enumerate(self.neuron_list):
			self.batches[idx % self.n_workers].append(neuron)
			self.batch_comm_idx[idx % self.n_workers].append(idx)
			print(f"Assigned Neuron {idx} to Worker {idx % self.n_workers}"	)

		self.barrier = Barrier(self.n_workers)

		# Start worker processes
		if self.n_workers == 1:
			SBDF_single(self.neuron_list, self.comm, self.writeStep, self.solver, self.er_scale, self.threads_per_process)
			return
		
		self.workers = []
		for wid, batch in enumerate(self.batches):
			p = Process(target=SBDF_worker,
						args=(wid, self.neuron_list, batch, self.comm, self.batch_comm_idx[wid], self.writeStep, self.solver, self.er_scale, self.barrier, self.threads_per_process))
			p.start()
			self.workers.append(p)

	def shutdown(self):
		if self.n_workers > 1:
			for p in self.workers:
				p.join(timeout=600)
		for neuron in self.neuron_list:
			neuron.close()
			neuron.unlink()

###########################################################

def SBDF_single(neuron_list, comm, writeStep, solver, er_scale, threads_per_process=1):

	from spine.utils.threading import apply_thread_limits_if_needed, set_num_threads
	set_num_threads(threads_per_process)
	apply_thread_limits_if_needed()

	# Limit to one thread per process
	from spine.utils.threading import detect_oversubscription
	if detect_oversubscription(verbose=True):
		print("[pyCalSim] Consider limiting to one thread for better performance with multiprocessing")

	# Initialize problem on neuron
	for neuron in neuron_list:
		gen_neuron(neuron)
		neuron.initialize_problem()
		neuron.initial_conditions()
		if neuron.model['voltage_coupling']:
			neuron.CableSettings.set_gating_variables()
			neuron.CableSettings.set_leak_reversal()
		
	# Attach to shared memory
	for neuron in neuron_list:
		neuron.shmC = shared_memory.SharedMemory(name=neuron.C_name)
		neuron.sol.C = np.ndarray(neuron.C_shape, dtype=neuron.C_dtype, buffer=neuron.shmC.buf)
		neuron.shmC0 = shared_memory.SharedMemory(name=neuron.C0_name)
		neuron.sol.C0 = np.ndarray(neuron.C0_shape, dtype=neuron.C0_dtype, buffer=neuron.shmC0.buf)
		neuron.shmV = shared_memory.SharedMemory(name=neuron.V_name)
		neuron.sol.V = np.ndarray(neuron.V_shape, dtype=neuron.V_dtype, buffer=neuron.shmV.buf)
		neuron.shmV0 = shared_memory.SharedMemory(name=neuron.V0_name)
		neuron.sol.V0 = np.ndarray(neuron.V0_shape, dtype=neuron.V0_dtype, buffer=neuron.shmV0.buf)

	# Initialize dicts
	neuronDict = [{} for _ in range(len(neuron_list))]

	# Loop over neurons
	for iter, neuron in enumerate(neuron_list):

		set_dictionary(neuron, neuronDict[iter], er_scale)
		set_states(neuron, neuronDict[iter])
		set_matrices(neuron, neuronDict[iter], solver)

	# SBDF2 solver
	t0 = 0.0
	for i in range(neuron_list[0].settings.ntime):
		for iter, neuron in enumerate(neuron_list):

			fluxes = SBDF_step(neuron, neuronDict[iter], comm, iter, solver, writeStep, i, t0)
			record_steps(neuron, neuronDict[iter], t0, fluxes=fluxes)

		if writeStep is not None and i % writeStep == 0:
			print(f'Time step {i} at time {t0:.2e} seconds completed.')
		elif i % int(neuron.settings.ntime/20) == 0:
			print(f'Time step {i} at time {t0:.2e} seconds completed.')
		t0 += neuron_list[0].settings.dt

		# Note: C0/V0 history update is handled by pointer swapping in SBDF_step()
		# via neuron.swap_history_arrays(). No explicit copy needed here.
	
	for neuron in neuron_list:
		if neuron.recorder['total_cyt'] and neuron.settings.output_folder is not None:
			np.savetxt(neuron.settings.top_folder+'results/'+neuron.settings.output_folder+'/total_cyt.txt', neuron.sol.total_cyt.reshape(1,-1))

		if neuron.recorder['total_er'] and neuron.settings.output_folder is not None:
			np.savetxt(neuron.settings.top_folder+'results/'+neuron.settings.output_folder+'/total_er.txt', neuron.sol.total_er.reshape(1,-1))

###########################################################

def SBDF_worker(worker_id, neuron_list, neuron_batch, comm, comm_idx, writeStep, solver, er_scale, barrier, barrier_timeout=300, threads_per_process=1):

	from spine.utils.threading import apply_thread_limits_if_needed, set_num_threads
	set_num_threads(threads_per_process)
	apply_thread_limits_if_needed()

	# Limit to one thread per process
	from spine.utils.threading import detect_oversubscription
	if detect_oversubscription(verbose=True):
		print("[pyCalSim] Consider limiting to one thread for better performance with multiprocessing")

	# Initialize problem on neuron
	for neuron in neuron_batch:
		gen_neuron(neuron)
		neuron.initialize_problem()
		neuron.initial_conditions()
		if neuron.model['voltage_coupling']:
			neuron.CableSettings.set_gating_variables()
			neuron.CableSettings.set_leak_reversal()
		
	# Attach to shared memory
	for neuron in neuron_list:
		neuron.shmC = shared_memory.SharedMemory(name=neuron.C_name)
		neuron.sol.C = np.ndarray(neuron.C_shape, dtype=neuron.C_dtype, buffer=neuron.shmC.buf)
		neuron.shmC0 = shared_memory.SharedMemory(name=neuron.C0_name)
		neuron.sol.C0 = np.ndarray(neuron.C0_shape, dtype=neuron.C0_dtype, buffer=neuron.shmC0.buf)
		neuron.shmV = shared_memory.SharedMemory(name=neuron.V_name)
		neuron.sol.V = np.ndarray(neuron.V_shape, dtype=neuron.V_dtype, buffer=neuron.shmV.buf)
		neuron.shmV0 = shared_memory.SharedMemory(name=neuron.V0_name)
		neuron.sol.V0 = np.ndarray(neuron.V0_shape, dtype=neuron.V0_dtype, buffer=neuron.shmV0.buf)

	# Initialize dicts
	neuronDict = [{} for _ in range(len(neuron_batch))]

	# Loop over neurons
	for iter, neuron in enumerate(neuron_batch):
		set_dictionary(neuron, neuronDict[iter], er_scale)
		set_states(neuron, neuronDict[iter])
		set_matrices(neuron, neuronDict[iter], solver)

	# SBDF2 solver
	t0 = 0.0
	for i in range(neuron_batch[0].settings.ntime):
		try:
			for iter, neuron in enumerate(neuron_batch):

				fluxes = SBDF_step(neuron, neuronDict[iter], comm, comm_idx[iter], solver, writeStep, i, t0)
				record_steps(neuron, neuronDict[iter], t0, fluxes=fluxes)

			if worker_id == 0 and (writeStep is not None and i % writeStep == 0):
				print(f'Time step {i} at time {t0:.2e} seconds completed.')
			elif worker_id == 0 and (i % int(neuron.settings.ntime/20) == 0):
				print(f'Time step {i} at time {t0:.2e} seconds completed.')
			t0 += neuron_batch[0].settings.dt

			try:
				barrier.wait(timeout=barrier_timeout)
			except mp.context.BrokenBarrierError:
				print(f"Worker {worker_id} timed out at step {i}. Exiting.")
				return

		except Exception as e:
			print(f"Worker {worker_id} encountered an exception: {e}")
			return
		
	for neuron in neuron_batch:
		if neuron.recorder['total_cyt'] and neuron.settings.output_folder is not None:
			np.savetxt(neuron.settings.top_folder+'results/'+neuron.settings.output_folder+'/total_cyt.txt', neuron.sol.total_cyt.reshape(1,-1))

		if neuron.recorder['total_er'] and neuron.settings.output_folder is not None:
			np.savetxt(neuron.settings.top_folder+'results/'+neuron.settings.output_folder+'/total_er.txt', neuron.sol.total_er.reshape(1,-1))

###########################################################

