import numpy as np

def _array_to_vtu_string(arr):
	"""Convert numpy array to space-separated string for VTU ASCII format.

	Optimized alternative to manual string concatenation in loops.
	"""
	return ' '.join(map(str, arr))

def vtu_write(neuron,t,tid):

	filename = neuron.settings.top_folder+'results/' + neuron.settings.output_folder + '/vtu/sol_t'+f"{tid:07d}"+'.vtu'

	# Open file once and write all content (avoid redundant open/close)
	f = open(filename,'w')
	f.write('<?xml version="1.0"?>')
	f.write('\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">')
	f.write('\n<Time timestep="' + str(round(t,5)) + '"/>')
	f.write('\n<UnstructuredGrid>')

	f.write('\n<FieldData>')
	f.write('\n<DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii" RangeMin="'+str(t)+'" RangeMax="'+str(t)+'">')
	f.write('\n'+str(t))
	f.write('\n</DataArray>')
	f.write('\n</FieldData>')
	
	f.write('\n<Piece NumberOfPoints="' + str(len(neuron.graph.nid)) + '" NumberOfCells="' + str(len(neuron.graph.nid)-1) + '">')
	
	f.write('\n<Points>')
	f.write('\n<DataArray type="Float32" NumberOfComponents="3" format="ascii">')
	coord_string = _array_to_vtu_string(neuron.graph.coord.flatten())
	f.write('\n' + coord_string)
	f.write('\n</DataArray>')
	f.write('\n</Points>')

	f.write('\n<Cells>')

	f.write('\n<DataArray type="Int32" Name="connectivity" format="ascii">')
	# Create connectivity by interleaving pid and nid (skip first node which is root)
	# Vectorized version: stack arrays and flatten
	connectivity = np.column_stack([neuron.graph.pid[1:], neuron.graph.nid[1:]]).flatten()
	edge_string = _array_to_vtu_string(connectivity)
	f.write('\n' + edge_string)
	f.write('\n</DataArray>')

	f.write('\n<DataArray type="Int32" Name="offsets" format="ascii">')
	# Offsets: 2, 4, 6, 8, ... (two vertices per cell)
	# Vectorized version: use numpy arange
	offsets = np.arange(2, 2*len(neuron.graph.nid), 2)
	offset_string = _array_to_vtu_string(offsets)
	f.write('\n' + offset_string)
	f.write('\n</DataArray>')

	f.write('\n<DataArray type="Int8" Name="types" format="ascii">')
	# Type 3 = VTK_LINE for all cells
	type_string = _array_to_vtu_string([3] * (len(neuron.graph.nid) - 1))
	f.write('\n' + type_string)
	f.write('\n</DataArray>')

	f.write('\n</Cells>')

	f.write('\n<PointData>')

	# Cytosolic calcium (conditional on vtu_cyt flag)
	if neuron.recorder.get('vtu_cyt', True):
		f.write('\n<DataArray type="Float32" Name="CYT" NumberOfComponents="1" format="ascii">')
		f.write('\n' + _array_to_vtu_string(neuron.sol.C))
		f.write('\n</DataArray>')

	# ER calcium (conditional on vtu_er flag)
	if neuron.recorder.get('vtu_er', True):
		f.write('\n<DataArray type="Float32" Name="ER" NumberOfComponents="1" format="ascii">')
		f.write('\n' + _array_to_vtu_string(neuron.sol.CE))
		f.write('\n</DataArray>')

	# Calcium buffer (conditional on vtu_calb flag)
	if neuron.recorder.get('vtu_calb', True):
		f.write('\n<DataArray type="Float32" Name="CALB" NumberOfComponents="1" format="ascii">')
		f.write('\n' + _array_to_vtu_string(neuron.sol.B))
		f.write('\n</DataArray>')

	# IP3 (conditional on IP3 model enabled AND vtu_ip3 flag)
	if neuron.model['IP3'] and neuron.recorder.get('vtu_ip3', True):
		f.write('\n<DataArray type="Float32" Name="IP3" NumberOfComponents="1" format="ascii">')
		f.write('\n' + _array_to_vtu_string(neuron.sol.IP3))
		f.write('\n</DataArray>')

	# Voltage (conditional on vtu_volt flag)
	if neuron.recorder.get('vtu_volt', True):
		f.write('\n<DataArray type="Float32" Name="VOLT" NumberOfComponents="1" format="ascii">')
		f.write('\n' + _array_to_vtu_string(neuron.sol.V))
		f.write('\n</DataArray>')

	# Radius (conditional on vtu_radius flag)
	if neuron.recorder.get('vtu_radius', True):
		f.write('\n<DataArray type="Float32" Name="RADIUS" NumberOfComponents="1" format="ascii">')
		f.write('\n' + _array_to_vtu_string(neuron.graph.radius))
		f.write('\n</DataArray>')

	# Integrated cytosolic calcium (conditional on recording enabled AND vtu_total_cyt flag)
	if neuron.recorder.get('total_cyt', False) and neuron.recorder.get('vtu_total_cyt', True):
		f.write('\n<DataArray type="Float32" Name="TOTAL_CYT" NumberOfComponents="1" format="ascii">')
		f.write('\n' + _array_to_vtu_string(neuron.sol.total_cyt))
		f.write('\n</DataArray>')

	# Integrated ER calcium (conditional on recording enabled AND vtu_total_er flag)
	if neuron.recorder.get('total_er', False) and neuron.recorder.get('vtu_total_er', True):
		f.write('\n<DataArray type="Float32" Name="TOTAL_ER" NumberOfComponents="1" format="ascii">')
		f.write('\n' + _array_to_vtu_string(neuron.sol.total_er))
		f.write('\n</DataArray>')

	f.write('\n</PointData>')
	f.write('\n</Piece>')
	f.write('\n</UnstructuredGrid>')
	f.write('\n</VTKFile>')

	f.close()
