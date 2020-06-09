#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# distutils: language = c++
# distutils: sources = RJJ_ObjGen_CreateObjs.cpp RJJ_ObjGen_MemManage.cpp RJJ_ObjGen_DetectDefn.cpp RJJ_ObjGen_ThreshObjs.cpp RJJ_ObjGen_Dmetric.cpp

# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

cimport numpy as np
import numpy as np
from libcpp.vector cimport vector

cdef extern from "RJJ_ObjGen.h":
	cdef long int CreateObjects( float * data_vals, long int * flag_vals,
							int size_x, int size_y, int size_z,
							int chunk_x_start, int chunk_y_start, int chunk_z_start,
							int radiusX, int radiusY, int radiusZ,
							int minSizeX, int minSizeY, int minSizeZ,
							int maxSizeX, int maxSizeY, int maxSizeZ,
							int minVoxels, int maxVoxels,
							float minIntens, float maxIntens,
							long int flag_value,
							long int start_obj, vector[object_props *] & detections, vector[long int] & obj_ids, vector[long int] & check_obj_ids, int obj_limit,
							int max_x_val, int max_y_val, int max_z_val,
							int ss_mode,
							size_t * data_metric, int * xyz_order)

	cdef void InitObjGen(vector[object_props *] & detections, long int & NOobj, int obj_limit, vector[long int] & obj_ids, vector[long int] & check_obj_ids, size_t *& data_metric, int *& xyz_order)
	cdef void FreeObjGen(vector[object_props *] & detections, size_t *& data_metric, int *& xyz_order)
	cdef void ThresholdObjs(vector[object_props *] & detections, long int NOobj, int obj_limit, int minSizeX, int minSizeY, int minSizeZ, int maxSizeX, int maxSizeY, int maxSizeZ, int minVoxels, int maxVoxels, int minLOS, int maxLOS, float minFill, float maxFill, float minIntens, float maxIntens)
	cdef void CreateMetric( size_t * data_metric, int * xyz_order, int size_x, int size_y, int size_z)

	cdef cppclass object_props:

		void CalcProps()

		# Bounding box
		int GetRAmin()
		int GetRAmax()
		int GetDECmin()
		int GetDECmax()
		int GetFREQmin()
		int GetFREQmax()

		# Center of mass
		float GetRAi()
		float GetDECi()
		float GetFREQi()

		float GetRAi_p()
		float GetDECi_p()
		float GetFREQi_p()

		float GetRAi_n()
		float GetDECi_n()
		float GetFREQi_n()

		# Geometric center
		float GetRA()
		float GetDEC()
		float GetFREQ()

		# Intensity extrema
		float GetMinI()
		float GetMaxI()

		# Total intensity
		float GetTI()
		float GetTI_p()
		float GetTI_n()

		# Intensity stats
		float GetAvgI()
		float GetSigmaI()
		float GetRMSI()

		# widths
		float Get_w20_min()
		float Get_w20_max()
		float Get_cw20_min()
		float Get_cw20_max()
		float Get_w50_min()
		float Get_w50_max()
		float Get_cw50_min()
		float Get_cw50_max()

		# Number of voxels
		int ShowVoxels()

		# Sparse representation functions
		int Get_srep_size(int index)
		int Get_srep_grid(int index)
		int Get_srep_strings(int index)

		# number of lines of sight that this object extends over
		int Get_LoScount()

def link_objects(data, objects, mask, radiusX = 0, radiusY = 0, radiusZ = 0, minSizeX = 1, minSizeY = 1, minSizeZ = 1, maxSizeX = -1, maxSizeY = -1, maxSizeZ = -1, minVoxels = 1, maxVoxels = -1, minLOS = 1, maxLOS = -1, minFill = -1, maxFill = 2, minIntens = -9E30, maxIntens = 9E30):
	"""
	Given a data cube and a binary mask, create a labeled version of the mask.
	In addition, close groups of objects can be linked together, so they have the same label.


	Parameters
	----------

	data : array
		The data

	objects: array
		The existing list of objects which will have new detections appended to it

	mask : array
		The binary mask

	radiusX, radiusY, radiusZ : int
		The merging length in all three dimensions

	minSizeX, minSizeY, minSizeZ : int
		The minimum size objects can have in all three dimensions

	maxSizeX, maxSizeY, maxSizeZ : int
		The minimum size objects can have in all three dimensions

	minVoxels, maxVoxels : int
		The minimum and maximum number of voxels comprising an object

	minLOS, maxLOS : int
		The mininum and maximum pixel-extent in the spatial (x,y) domain of the data a source must have

	minFill, maxFill : float
		The minimum and maximum percentage of the voxels within an object's bounding box that is flagged as source

	ss_mode : int
		The linking method. A value of 1 uses a cuboid and all other values use an elliptical cylinder.


	Returns
	-------

	objects : list
		Lists of lists. Order of parameters:
			Geometric Center X,Y,Z
			Center-Of-Mass X,Y,X
			Bounding Box Xmin, Xmax, Ymin, Ymax, Zmin, Zmax
			Flux Min, Flux Max, Flux Total,

		The Bounding box are defined is such a way that they can be
		used as slices, i.e. data[Zmin:Zmax]

	mask : array
		The labeled and linked integer mask
	"""
	try:
		objects
	except:
		objects = []
	return _link_objects(data.astype(np.single, copy = False), objects, mask.astype(np.int_, copy = False), radiusX, radiusY, radiusZ, minSizeX, minSizeY, minSizeZ, maxSizeX, maxSizeY, maxSizeZ, minVoxels, maxVoxels, minLOS, maxLOS, minFill, maxFill, minIntens, maxIntens)

cdef _link_objects(np.ndarray[dtype = float, ndim = 3] data, objects, np.ndarray[dtype = long int, ndim = 3] mask,
				   int radiusX = 3, int radiusY = 3, int radiusZ = 5,
				   int minSizeX = 1, int minSizeY = 1, int minSizeZ = 1,
				   int maxSizeX = -1, int maxSizeY = -1, int maxSizeZ = -1,
				   int minVoxels = 1, int maxVoxels = -1,
				   int minLOS = 1, int maxLOS = -99, float minFill = -1, float maxFill = 2,
				   float minIntens = -9E30, float maxIntens = 9E30):

	cdef int i, x, y, z, g, g_start, g_end
	cdef long int obj_id
	cdef int obj_batch

	cdef int size_x = data.shape[2]
	cdef int size_y = data.shape[1]
	cdef int size_z = data.shape[0]

	# set the number of existing objects to be the starting id
	try:
		obj_id = objects.shape[0]
	except:
		obj_id = 0

	# Convert binary mask to conform with the object code
	for z in range(size_z):
		for y in range(size_y):
			for x in range(size_x):
				if mask[z,y,x] > 0:
					mask[z,y,x] = -1
				else:
					mask[z,y,x] = -99

	# Define arrays storing datacube geometry metric
	cdef size_t * data_metric
	cdef int * xyz_order

	# Chunking is disabled for this interface
	cdef int chunk_x_start = 0
	cdef int chunk_y_start = 0
	cdef int chunk_z_start = 0

	# Define value that is used to mark sources in the mask
	cdef long int flag_val = -1

	# Specify size of allocated object groups
	cdef int obj_limit = 1000

	# Object and ID arrays; will be written to by the function
	cdef vector[object_props *] detections
	cdef vector[long int] obj_ids
	cdef vector[long int] check_obj_ids
	cdef long int NOobj = 0

	# Define linking style: 1 for Rectangle, else ellipse
	cdef int ss_mode = 0

	# Inititalize object pointers
	InitObjGen(detections, NOobj, obj_limit, obj_ids, check_obj_ids, data_metric, xyz_order)

	# Define the mapping of RA, Dec and frequency to the datacube's first three axes. RA, Dec and freq. --> 1, 2, 3: use the default values of x=RA=1 y=Dec=2 z=freq.=3
	xyz_order[0] = 1
	xyz_order[1] = 2
	xyz_order[2] = 3

	# create metric for accessing this data chunk in arbitrary x,y,z order
	CreateMetric(data_metric, xyz_order, size_x, size_y, size_z)

	# Create and threshold objects
	NOobj = CreateObjects(<float *> data.data, <long int *> mask.data, size_x, size_y, size_z, chunk_x_start, chunk_y_start, chunk_z_start, radiusX, radiusY, radiusZ, minSizeX, minSizeY, minSizeZ, maxSizeX, maxSizeY, maxSizeZ, minVoxels, maxVoxels, minIntens, maxIntens, flag_val, obj_id, detections, obj_ids, check_obj_ids, obj_limit, size_x, size_y, size_z, ss_mode, data_metric, xyz_order)
	ThresholdObjs(detections, NOobj, obj_limit, minSizeX, minSizeY, minSizeZ, maxSizeX, maxSizeY, maxSizeZ, minVoxels, maxVoxels, minLOS, maxLOS, minFill, maxFill, minIntens, maxIntens)

	# Reset output mask
	for z in range(size_z):
		for y in range(size_y):
			for x in range(size_x):
				mask[z,y,x] = 0

	# Create Python list `objects' from C++ vector `detections' and re-label mask with final, sequential IDs
	for i in range(NOobj):
		# calculate batch number for this object --- which group of objects does it belong to
		obj_batch = i / obj_limit

		if detections[obj_batch][i - (obj_batch * obj_limit)].ShowVoxels() >= 1:
			obj_id += 1

			obj = []
			obj.append(obj_id)

			detections[obj_batch][i - (obj_batch * obj_limit)].CalcProps()

			# Geometric center
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetRA())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetDEC())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetFREQ())

			# Center of mass
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetRAi())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetDECi())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetFREQi())

			# Bounding box values
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetRAmin())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetRAmax())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetDECmin())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetDECmax())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetFREQmin())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetFREQmax())

			# Number of voxels
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].ShowVoxels())

			# Min/Max/Total flux
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetMinI())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetMaxI())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetTI())

			### new properties added for reliability analyses improvements

			# Positive and negative centers of mass
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetRAi_p())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetDECi_p())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetFREQi_p())

			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetRAi_n())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetDECi_n())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetFREQi_n())

			# Positive and negative total flux
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetTI_p())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetTI_n())

			# Mean, std.dev. and RMS flux
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetAvgI())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetSigmaI())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetRMSI())

			# W20 and W50
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].Get_w20_max() - detections[obj_batch][i - (obj_batch * obj_limit)].Get_w20_min())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].Get_w50_max() - detections[obj_batch][i - (obj_batch * obj_limit)].Get_w50_min())

			# C.F.D. W20 and C.F.D. W50
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].Get_cw20_max() - detections[obj_batch][i - (obj_batch * obj_limit)].Get_cw20_min())
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].Get_cw50_max() - detections[obj_batch][i - (obj_batch * obj_limit)].Get_cw50_min())

			# the sizes of the bounding box
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetRAmax() - detections[obj_batch][i - (obj_batch * obj_limit)].GetRAmin() + 1);
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetDECmax() - detections[obj_batch][i - (obj_batch * obj_limit)].GetDECmin() + 1);
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].GetFREQmax() - detections[obj_batch][i - (obj_batch * obj_limit)].GetFREQmin() + 1);

			# the number of lines of sight that the object covers
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].Get_LoScount());

			# the fill factor of the object's bounding box
			obj.append(detections[obj_batch][i - (obj_batch * obj_limit)].ShowVoxels()/((detections[obj_batch][i - (obj_batch * obj_limit)].GetRAmax() - detections[obj_batch][i - (obj_batch * obj_limit)].GetRAmin() + 1)*(detections[obj_batch][i - (obj_batch * obj_limit)].GetDECmax() - detections[obj_batch][i - (obj_batch * obj_limit)].GetDECmin() + 1)*(detections[obj_batch][i - (obj_batch * obj_limit)].GetFREQmax() - detections[obj_batch][i - (obj_batch * obj_limit)].GetFREQmin() + 1)));

			objects.append(obj)

			for y in range(detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(2), detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(3) + 1):
				for x in range(detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(0), detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(1) + 1):
					g_start = detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_grid(((((y - detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(2)) * (detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(1) - detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(0) + 1)) + x - detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(0))))
					g_end = detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_grid(((((y - detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(2)) * (detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(1) - detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(0) + 1)) + x - detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_size(0) + 1)))

					for g in range(g_start, g_end):
						mask[detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_strings((2 * g)) : detections[obj_batch][i - (obj_batch * obj_limit)].Get_srep_strings((2 * g) + 1) + 1, y, x] = obj_id

	# Free memory for object pointers
	FreeObjGen(detections, data_metric, xyz_order)

	return objects, mask
