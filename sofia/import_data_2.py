#! /usr/bin/env python

import os
import re
import math
import numpy as np
from astropy.io import fits
from sofia import error as err


# ==========================
# FUNCTION: Import data cube
# ==========================

def import_data(doSubcube, inFile, weightsFile, maskFile, weightsFunction=None, subcube=[], subcubeMode="pixel", doFlag=False, flagRegions=False, flagFile="", cubeOnly=False):
	# Basic sanity checks on user input
	err.ensure(
		os.path.isfile(inFile),
		"Data file not found:\n  " + str(inFile))
	
	# -------------------------------
	# Open input cube and read header
	# -------------------------------
	err.message("Loading input data cube.")
	try:
		f = fits.open(inFile, mode="readonly", memmap=False, do_not_scale_image_data=False)
		header = f[0].header
	except:
		err.error("Failed to load primary HDU of data file:\n  " + str(inFile))
	
	# Extract axis sizes and types
	n_axes, axis_size, axis_type = extract_axis_size(header)
	
	# Check dimensionality of data cube
	check_cube_dimensions(n_axes, axis_size, cube_name="data cube")
	
	# Print some information
	err.message("  Data cube has {0:d} axes.".format(header["NAXIS"]))
	err.message("    Types: " + str(axis_type))
	err.message("    Sizes: " + str(axis_size))
	
	# Extract subcube boundaries if requested
	if len(subcube): subcube = get_subcube_range(header, n_axes, axis_size, subcube, subcubeMode)
	else: subcube = []
	
	# --------------------------------
	# Read requested subregion of data
	# --------------------------------
	# 2-D image
	if n_axes == 2:
		fullshape = [axis_size[1], axis_size[0]]
		if len(subcube):
			data = np.array([f[0].section[subcube[2]:subcube[3], subcube[0]:subcube[1]]])
			header["CRPIX1"] -= subcube[0]
			header["CRPIX2"] -= subcube[2]
			header["NAXIS1"] = subcube[1] - subcube[0]
			header["NAXIS2"] = subcube[3] - subcube[2]
		else:
			data = np.array([f[0].data])
	
	# 3-D cube
	elif n_axes == 3:
		fullshape = [axis_size[2], axis_size[1], axis_size[0]]
		if len(subcube):
			data = f[0].section[subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
			header["CRPIX1"] -= subcube[0]
			header["CRPIX2"] -= subcube[2]
			header["CRPIX3"] -= subcube[4]
			header["NAXIS1"] = subcube[1] - subcube[0]
			header["NAXIS2"] = subcube[3] - subcube[2]
			header["NAXIS3"] = subcube[5] - subcube[4]
		else:
			data = f[0].data
	
	#4-D hypercube
	else:
		fullshape = [axis_size[2], axis_size[1], axis_size[0]]
		if len(subcube):
			data = f[0].section[0, subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
			header["CRPIX1"] -= subcube[0]
			header["CRPIX2"] -= subcube[2]
			header["CRPIX3"] -= subcube[4]
			header["NAXIS1"] = subcube[1] - subcube[0]
			header["NAXIS2"] = subcube[3] - subcube[2]
			header["NAXIS3"] = subcube[5] - subcube[4]
		else:
			data = f[0].section[0]
	
	# Close input cube
	f.close()
	err.message("Input data cube loaded.")
	
	# ---------------------------------------------------------
	# If no additional actions required, return data and header
	# ---------------------------------------------------------
	if cubeOnly: return data, header
	
	# ---------------------------------------------------
	# Otherwise carry out additional actions as requested
	# ---------------------------------------------------
	# Weighting
	if weightsFile:
		data = apply_weights_file(data, weightsFile, subcube)
	elif weightsFunction:
		data = apply_weights_function(data, weightsFunction)
	
	# Flagging
	if doFlag:
		data = apply_flagging(data, flagFile, flagRegions, subcube)
	
	# Masking
	if maskFile:
		mask = import_mask(maskFile, header, axis_size, subcube)
	else:
		# Create an empty mask if none is provided.
		mask = np.zeros(data.shape, dtype=bool)
	
	return data, header, mask, subcube



# ============================================
# FUNCTION: Return the requested subcube range
# ============================================

def get_subcube_range(header, n_axes, axis_size, subcube, subcubeMode):
	# Basic sanity checks
	err.ensure(
		subcubeMode in {"pixel", "world"},
		"Subcube mode must be 'pixel' or 'world'.")
	err.ensure(
		(len(subcube) == 4 and n_axes == 2) or (len(subcube) == 6 and n_axes > 2),
		"Subcube range must contain 4 values for 2-D cubes\n"
		"or 6 values for 3-D/4-D cubes.")
	
	# -----------------
	# World coordinates
	# -----------------
	if subcubeMode == "world":
		# Read WCS information
		try:
			wcsin = wcs.WCS(header)
		except:
			err.error("Failed to read WCS information from data cube header.")
		
		# Calculate cos(dec) correction for RA range:
		if wcsin.wcs.cunit[0] == "deg" and wcsin.wcs.cunit[1] == "deg":
			corrfact = math.cos(math.radians(subcube[1]))
		
		if n_axes == 4:
			subcube = wcsin.wcs_world2pix(np.array([[subcube[0] - subcube[3] / corrfact, subcube[1] - subcube[4], subcube[2] - subcube[5], 0], [subcube[0] + subcube[3] / corrfact, subcube[1] + subcube[4], subcube[2] + subcube[5], 0]]), 0)[:, :3]
		elif n_axes == 3:
			subcube = wcsin.wcs_world2pix(np.array([[subcube[0] - subcube[3] / corrfact, subcube[1] - subcube[4], subcube[2] - subcube[5]], [subcube[0] + subcube[3] / corrfact, subcube[1] + subcube[4], subcube[2] + subcube[5]]]), 0)
		elif n_axes == 2:
			subcube = wcsin.wcs_world2pix(np.array([[subcube[0] - subcube[2] / corrfact, subcube[1] - subcube[3]], [subcube[0] + subcube[2] / corrfact, subcube[1] + subcube[3]]]), 0)
		else:
			err.error("Unsupported number of axes.")
		
		# Flatten array
		subcube = np.ravel(subcube, order="F")
		
		# Ensure that min pix coord < max pix coord for all axes.
		# This operation is meaningful because wcs_world2pix returns negative pixel coordinates
		# only for pixels located before an axis' start (i.e., negative pixel coordinates should
		# not be interpreted as counting backward from an axis' end).
		subcube[0], subcube[1] = correct_order(subcube[0], subcube[1])
		subcube[2], subcube[3] = correct_order(subcube[2], subcube[3])
		if len(subcube) == 6: subcube[4], subcube[5] = correct_order(subcube[4], subcube[5])
		
		# Convert to integer
		subcube = list(subcube.astype(int))
		
		# Constrain subcube to be within cube boundaries
		for axis in range(min(3, n_axes)):
			err.ensure(subcube[2 * axis + 1] >= 0 and subcube[2 * axis] < axis_size[axis],
				"Subcube outside input cube range for axis {0:d}.".format(axis))
			subcube[2 * axis] = max(subcube[2 * axis], 0)
			subcube[2 * axis + 1] = min(subcube[2 * axis + 1] + 1, axis_size[axis])
	
	# -----------------
	# Pixel coordinates
	# -----------------
	else:
		# Ensure that pixel coordinates are integers
		for value in subcube:
			err.ensure(type(value) == int, "Subcube boundaries must be integer values.")
		
		# Sanity checks on boundaries
		for axis in range(min(3, n_axes)):
			# Ensure correct order
			err.ensure(subcube[2 * axis] < subcube[2 * axis + 1],
				"Lower subcube boundary greater than upper boundary.\nPlease check your input.")
			# Adjust lower boundary
			subcube[2 * axis] = max(subcube[2 * axis], 0)
			subcube[2 * axis] = min(subcube[2 * axis], axis_size[axis] - 1)
			# Adjust upper boundary:
			subcube[2 * axis + 1] = max(subcube[2 * axis + 1], 1)
			subcube[2 * axis + 1] = min(subcube[2 * axis + 1], axis_size[axis])
	
	# Report final subcube boundaries
	err.message("  Loading subcube of range " + str(subcube) + '.')
	
	return subcube



# ============================
# FUNCTION: Apply weights file
# ============================

def apply_weights_file(data, weightsFile, subcube):
	# Load weights cube
	err.message("Applying weights cube:\n  " + str(weightsFile))
	try:
		f = fits.open(weightsFile, memmap=False)
		header_weights = f[0].header
	except:
		err.error("Failed to read weights cube.")
	
	# Extract axis sizes and types
	n_axes_weights, axis_size_weights, axis_type_weights = extract_axis_size(header_weights)
	
	# Ensure correct dimensionality
	check_cube_dimensions(n_axes_weights, axis_size_weights, cube_name="weights cube", min_dim=1, max_dim=4)
	
	# Multiply data by weights
	# 1-D spectrum
	if n_axes_weights == 1:
		err.warning("Weights cube has 1 axis; interpreted as spectrum.\nAdding first and second axis.")
		if len(subcube):
			err.ensure(len(subcube) == 6, "Subcube list must have 6 entries ({0:d} given).".format(len(subcube)))
			data *= np.reshape(f[0].section[subcube[4]:subcube[5]], (-1, 1, 1))
		else:
			data *= reshape(f[0].data, (-1, 1, 1))
	
	# 2-D image
	elif n_axes_weights == 2:
		if len(subcube) == 6 or len(subcube) == 4:
			data *= np.array([f[0].section[subcube[2]:subcube[3], subcube[0]:subcube[1]]])
		else:
			data *= np.array([f[0].data])
	
	# 3-D cube
	elif n_axes_weights == 3:
		if len(subcube) == 6:
			data *= f[0].section[subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
		else:
			data *= f[0].data
	
	# 4-D hypercube
	else:
		if len(subcube) == 6:
			data *= f[0].section[0, subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
		else:
			data *= f[0].section[0]
	
	f.close()
	err.message("  Weights cube applied.")
	
	return data



# ================================
# FUNCTION: Apply weights function
# ================================

def apply_weights_function(data, weightsFunction):
	err.message("Applying weights function:\n  " + str(weightsFunction))
	
	# Define whitelist of allowed character sequences and import relevant Numpy functions
	whitelist = ["x", "y", "z", "e", "E", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "exp", "log", "sqrt", "square", "power", "absolute", "fabs", "sign"]
	from numpy import sin, cos, tan, arcsin, arccos, arctan, arctan2, sinh, cosh, tanh, arcsinh, arccosh, arctanh, exp, log, sqrt, square, power, absolute, fabs, sign
	
	# Search for all keywords consisting of consecutive sequences of alphabetical characters
	keywordsFound = filter(None, re.split("[^a-zA-Z]+", str(weightsFunction)))
	
	# Check for non-whitelisted sequences
	for keyword in keywordsFound:
		err.ensure(keyword in whitelist,
			"Unknown keyword '" + str(keyword) + "' found in weights function:\n"
			"  " + str(weightsFunction) + "\n"
			"Please check your input.")
	
	# Loop over all channels
	for i in range(data.shape[0]):
		# Create index arrays over 2-D planes (i.e. of width dz = 1)
		z, y, x = np.indices((1, data.shape[1], data.shape[2]))
		z += i
		
		# Multiply each plane by weights function
		try:
			data[z, y, x] *= eval(str(weightsFunction))
			# NOTE: eval() should be safe now as we don't allow for non-whitelisted keywords.
		except:
			err.error(
				"Failed to evaluate weights function:\n"
				"  " + str(weightsFunction) + "\n"
				"Please check your input.")
	
	err.message("  Weights function applied.")
	
	return data



# ========================
# FUNCTION: Apply flagging
# ========================

def apply_flagging(data, flagFile, flagRegions, subcube):
	# -------------------------------
	# Apply flagging cube if provided
	# -------------------------------
	if flagFile:
		err.message("Applying flagging cube:\n  " + str(flagFile))
		
		try:
			f = fits.open(flagFile, memmap=False)
			header_flags = f[0].header
		except:
			err.error("Failed to read flagging cube.")
			
		# Extract axis sizes and types
		n_axes_flags, axis_size_flags, axis_type_flags = extract_axis_size(header_flags)
		
		# Ensure correct dimensionality
		check_cube_dimensions(n_axes_flags, axis_size_flags, cube_name="flagging cube")
		
		# Apply flagging
		# 2-D image
		if n_axes_flags == 2:
			for chan in range(data.shape[0]):
				if len(subcube) == 6 or len(subcube) == 4:
					data[chan][np.isnan(f[0].section[subcube[2]:subcube[3], subcube[0]:subcube[1]])] = np.nan
				else:
					data[chan][np.isnan(np.array([f[0].data]))] = np.nan
		
		# 3-D cube
		elif n_axes_flags == 3:
			if len(subcube) == 6:
				data[np.isnan(f[0].section[subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]])] = np.nan
			else:
				data[np.isnan(f[0].data)] = np.nan
		
		# 4-D hypercube
		else:
			if len(subcube) == 6:
				data[np.isnan(f[0].section[0, subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]])] = np.nan
			else:
				data[np.isnan(f[0].section[0])] = np.nan
		
		f.close()
		err.message("Flagging cube applied.")
	
	# ----------------------------------
	# Apply flagging regions if provided
	# ----------------------------------
	if flagRegions:
		err.message("Applying flagging regions:\n  " + str(flagRegions))
		dim = len(data.shape)
		
		try:
			for region in flagRegions:
				for i in range(0, len(region) / 2):
					if region[2 * i + 1] == "":
						region[2 * i + 1] = data.shape[dim - i - 1]
				if len(region) == 2:
					data[0, region[2]:region[3], region[0]:region[1]] = np.nan
				else:
					data[region[4]:region[5], region[2]:region[3], region[0]:region[1]] = np.nan
			err.message("Flagging regions applied.")
		except:
			err.error("Flagging did not succeed. Please check the dimensions\nof your data cube and flagging regions.")
	
	return data



# ========================
# FUNCTION: Read mask file
# ========================

def import_mask(maskFile, header, axis_size, subcube):
	err.message("Loading mask cube:\n  " + str(maskFile))
	
	try:
		f = fits.open(maskFile, memmap=False)
		header_mask = f[0].header
	except:
		err.error("Failed to read mask cube.")
	
	# Extract axis sizes and types
	n_axes_mask, axis_size_mask, axis_type_mask = extract_axis_size(header_mask)
	
	# Ensure correct dimensionality
	check_cube_dimensions(n_axes_mask, axis_size_mask, cube_name="mask cube", min_dim = 1, max_dim = 4)
	
	# 1-D spectrum
	if n_axes_mask == 1:
		err.warning("Mask cube has 1 axis; interpreted as spectrum.\nAdding first and second axis.")
		if header_mask['CRVAL1'] != header['CRVAL1']: err.warning("Input cube and mask may not be on the same WCS grid.")
		
		if len(subcube) == 6:
			if header_mask["NAXIS1"] == axis_size[2]:
				err.message("  Input mask cube already matches size of data subcube.\n  No subcube selection applied.")
				mask = np.reshape(f[0].data, (-1, 1, 1))
			elif header_mask["NAXIS1"] == fullshape[0]:
				err.message("  Subcube selection applied to input mask cube.")
				mask = np.reshape(f[0].section[subcube[4]:subcube[5]], (-1, 1, 1))
			else:
				err.error("Data subcube does not match size of mask subcube or full mask.")
		elif not len(subcube):
			mask = np.reshape(f[0].data, (-1, 1, 1))
		else:
			err.error("The subcube list must have 6 entries ({0:d} given).".format(len(subcube)))
	
	# 2-D image
	elif n_axes_mask == 2:
		if header_mask["CRVAL1"] != header["CRVAL1"] or header_mask["CRVAL2"] != header["CRVAL2"]: err.warning("Input cube and mask may not be on the same WCS grid.")
		
		if len(subcube) == 6 or len(subcube) == 4:
			if header_mask["NAXIS1"] == axis_size[0] and header_mask["NAXIS2"] == axis_size[1]:
				err.message("  Input mask cube already matches size of data subcube.\n  No subcube selection applied.")
				mask = np.array([f[0].data])
			elif header_mask["NAXIS1"] == fullshape[2] and header_mask["NAXIS2"] == fullshape[1]:
				err.message("  Subcube selection applied to input mask cube.")
				mask = np.array([f[0].section[subcube[2]:subcube[3], subcube[0]:subcube[1]]])
			else:
				err.error("Data subcube does not match size of mask subcube or full mask.")
		else: mask = np.array([f[0].data])
	
	# 3-D cube
	elif n_axes_mask == 3:
		if header_mask["CRVAL1"] != header["CRVAL1"] or header_mask["CRVAL2"] != header["CRVAL2"] or header_mask["CRVAL3"] != header["CRVAL3"]: err.warning("Input cube and mask may not be on the same WCS grid.")
		
		if len(subcube) == 6:
			if header_mask["NAXIS1"] == axis_size[0] and header_mask["NAXIS2"] == axis_size[1] and header_mask["NAXIS3"] == axis_size[2]:
				err.message("  Input mask cube already matches size of data subcube.\n  No subcube selection applied.")
				mask = f[0].data
			elif header_mask["NAXIS1"] == fullshape[2] and header_mask["NAXIS2"] == fullshape[1] and header_mask["NAXIS3"] == fullshape[0]:
				err.message("  Subcube selection applied to input mask cube.")
				mask = f[0].section[subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
			else:
				err.error("Data subcube does not match size of mask subcube or full mask.")
		else: mask = f[0].data
	
	# 4-D hypercube
	else:
		if header_mask["CRVAL1"] != header["CRVAL1"] or header_mask["CRVAL2"] != header["CRVAL2"] or header_mask["CRVAL3"] != header["CRVAL3"]: err.warning("Input cube and mask may not be on the same WCS grid.")
		
		if len(subcube) == 6:
			if header_mask["NAXIS1"] == axis_size[0] and header_mask["NAXIS2"] == axis_size[1] and header_mask["NAXIS3"] == axis_size[2]:
				err.message("  Input mask cube already matches size of data subcube.\n  No subcube selection applied.")
				mask = f[0].section[0]
			elif header_mask["NAXIS1"] == fullshape[2] and header_mask["NAXIS2"] == fullshape[1] and header_mask["NAXIS3"] == fullshape[0]:
				err.message("  Subcube selection applied to input mask cube.")
				mask = f[0].section[0, subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
			else:
				err.error("Data subcube does not match size of mask subcube or full mask.")
		else: mask = f[0].section[0]
	
	mask[mask > 0] = 1
	f.close()
	err.message("Mask cube loaded.")
	
	# In all cases, convert mask to Boolean with masked pixels set to 1.
	return (mask > 0).astype(bool)



# ==================================
# FUNCTION: Swap order of two values
# ==================================

def correct_order(a, b):
	if a > b: return b, a
	return a, b



# ===============================
# FUNCTION: Check cube dimensions
# ===============================

def check_cube_dimensions(n_axes, axis_size, cube_name="data cube", min_dim=2, max_dim=4):
	err.ensure(
		n_axes >= min_dim and n_axes <= max_dim,
		str(cube_name).capitalize() + " must have {0:d} to {1:d} dimensions.".format(min_dim, max_dim))
	err.ensure(
		n_axes != 4 or axis_size[3] == 1,
		"Size of 4th axis of " + str(cube_name) + " is > 1. 4-D cubes can\n"
		"only be processed if 4th axis has size 1.")
	return



# ==============================================
# FUNCTION: Extract axis sizes/types from header
# ==============================================

def extract_axis_size(header):
	n_axes = int(header["NAXIS"])
	axis_size = []
	axis_type = []
	for i in range(n_axes):
		axis_size.append(int(header["NAXIS{0:d}".format(i + 1)]))
		axis_type.append(str(header["CTYPE{0:d}".format(i + 1)]))
	return n_axes, axis_size, axis_type
