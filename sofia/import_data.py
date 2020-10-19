#! /usr/bin/env python

# import default python libraries
from astropy.io import fits
import os
import sys
from numpy import *
import re
from sofia import error as err


def read_data(doSubcube, inFile, invertData, weightsFile, maskFile, sources, weightsFunction = None, subcube=[], subcubeMode="pixel", doFlag=False, flagRegions=False, flagFile="", cubeOnly=False):
	# import the fits file into an numpy array for the cube and a dictionary for the header:
	# the data cube is converted into a 3D array
	
	# First check if the data file exists:
	if not os.path.isfile(inFile):
		sys.stderr.write("ERROR: The specified data cube does not exist.\n       Cannot find: " + inFile + "\n")
		raise SystemExit(1)
	
	# Handle sub-cube if requested by user:
	# ALERT: Note that subcube boundaries for any axis are interpreted as [min, max) rather than [min, max] as expected by the user!!!
	#        This should be changed to avoid confusion. In addition, we should agree at some point whether SoFiA should be 0 or 1-based.
	if doSubcube:
		# Ensure that sub-cube specifications are as expected:
		if len(subcube) not in [4, 6] or subcubeMode not in ["pixel", "world"]:
			sys.stderr.write("ERROR: import.subcubeMode can only be 'pixel' or 'world',\n       and import.subcube must have 4 or 6 entries.\n")
			raise SystemExit(1)
		
		# Read the file header:
		from astropy import wcs
		hdulist = fits.open(inFile, memmap=False)
		header = hdulist[0].header
		hdulist.close()
		
		# Extract cube size information:
		axisSize = []
		for axis in range(min(3, header["NAXIS"])):
			axisSize.append(int(header["NAXIS%i" % (axis + 1)]))
		
		# Sub-cube in world coordinates:
		if subcubeMode == "world":
			sys.stdout.write("Calculating subcube boundaries from input WCS centre and radius\n")
			
			# Read WCS information:
			try:
				wcsin = wcs.WCS(header)
			except:
				sys.stderr.write("ERROR: Failed to read WCS information from data cube header.\n")
				raise SystemExit(1)
			
			# Calculate cos(Dec) correction for RA range:
			if wcsin.wcs.cunit[0] == "deg" and wcsin.wcs.cunit[1] == "deg":
				corrfact = cos(subcube[1] / 180.0 * pi)
			
			if header["NAXIS"] == 4:
				subcube = wcsin.wcs_world2pix(array([[subcube[0] - subcube[3] / corrfact, subcube[1] - subcube[4], subcube[2] - subcube[5], 0], [subcube[0] + subcube[3] / corrfact, subcube[1] + subcube[4], subcube[2] + subcube[5], 0]]), 0)[:,:3]
			elif header["NAXIS"] == 3:
				subcube = wcsin.wcs_world2pix(array([[subcube[0] - subcube[3] / corrfact, subcube[1] - subcube[4], subcube[2] - subcube[5]], [subcube[0] + subcube[3] / corrfact, subcube[1] + subcube[4], subcube[2] + subcube[5]]]), 0)
			elif header["NAXIS"] == 2:
				subcube = wcsin.wcs_world2pix(array([[subcube[0] - subcube[2] / corrfact, subcube[1] - subcube[3]], [subcube[0] + subcube[2] / corrfact, subcube[1] + subcube[3]]]), 0)
			else:
				sys.stderr.write("ERROR: Unsupported number of axes.\n")
				raise SystemExit(1)
			
			subcube = ravel(subcube, order="F")
			# make sure min pix coord is < max pix coord for all axes
			# this operation is meaningful because wcs_world2pix returns negative pixel coordinates only for pixels located before an axis" start
			# (i.e., negative pixel coordinates should not be interpreted as counting backward from an axis' end)
			if subcube[0] > subcube[1]:
				subcube[0], subcube[1] = subcube[1], subcube[0]
			if subcube[2] > subcube[3]:
				subcube[2], subcube[3] = subcube[3], subcube[2]
			if len(subcube) == 6:
				if subcube[4] > subcube[5]:
					subcube[4], subcube[5] = subcube[5], subcube[4]
			# constrain subcube to be within the cube boundaries; if this is not possible then exit
			for axis in range(min(3, header["NAXIS"])):
				if ceil(subcube[1 + 2 * axis]) < 0 or floor(subcube[2 * axis]) >= header["NAXIS%i" % (axis + 1)]:
					sys.stderr.write("ERROR: The requested subcube is outside the input cube along axis %i \n" % (axis))
					raise SystemExit(1)
				else:
					subcube[2 * axis] = max(0, floor(subcube[2 * axis]))
					subcube[1 + 2 * axis] = min(header["NAXIS%i" % (axis + 1)] - 1, ceil(subcube[1 + 2 * axis])) + 1
			subcube = list(subcube.astype(int))
		
		# Sub-cube in pixel coordinates:
		else:
			# Ensure that pixel coordinates are integers:
			for ss in subcube:
				if type(ss) != int:
					sys.stderr.write("ERROR: For subcubeMode = pixel, subcube boundaries must be integer.\n")
					sys.stderr.write("       The %i-th coordinate is not an integer value.\n" % subcube.index(ss))
					raise SystemExit(1)
			
			# Ensure to be within cube boundaries:
			for axis in range(min(3, header["NAXIS"])):
				# Lower boundary:
				if subcube[2 * axis] < 0:
					subcube[2 * axis] = 0
					sys.stderr.write("WARNING: Adjusting lower subcube boundary to 0 for axis %i.\n" % (axis + 1))
				elif subcube[2 * axis] >= axisSize[axis]:
					subcube[2 * axis] = axisSize[axis] - 1
					sys.stderr.write("WARNING: Adjusting lower subcube boundary to %i for axis %i.\n" % (axisSize[axis] - 1, axis + 1))
				# Upper boundary:
				if subcube[2 * axis + 1] < 1:
					subcube[2 * axis + 1] = 1
					sys.stderr.write("WARNING: Adjusting upper subcube boundary to 1 for axis %i.\n" % (axis + 1))
				elif subcube[2 * axis + 1] > axisSize[axis]:
					subcube[2 * axis + 1] = axisSize[axis]
					sys.stderr.write("WARNING: Adjusting upper subcube boundary to %i for axis %i.\n" % (axisSize[axis], axis + 1))
			
			# Ensure that boundaries are internally consistent:
			for axis in range(min(3, header["NAXIS"])):
				if subcube[2 * axis] >= subcube[2 * axis + 1]:
					sys.stderr.write("ERROR: Lower subcube boundary greater than upper subcube boundary.\n")
					sys.stderr.write("       Please check your input.\n")
					raise SystemExit(1)
		
		# Report final subcube boundaries:
		if len(subcube) == 4:
			sys.stdout.write("Loading subcube of " + str(inFile) + "\n  defined by [x1 x2 y1 y2] = " + str(subcube) + "\n")
		else:
			sys.stdout.write("Loading subcube of " + str(inFile) + "\n  defined by [x1 x2 y1 y2 z1 z2] = " + str(subcube) + "\n")
	else:
		sys.stdout.write("Loading cube " + str(inFile) + "\n")
		subcube = []
	
	# Open FITS file:
	try:
		f = fits.open(inFile, memmap=False)
		dict_Header = f[0].header
	except:
		sys.stderr.write("ERROR: Failed to load primary HDU of FITS file " + str(inFile) + "\n")
		raise SystemExit(1)
	
	# Check whether the number of dimensions is acceptable and read data accordingly.
	# The default is three axes:
	if dict_Header["NAXIS"] == 3:
		print ("The input cube has 3 axes:")
		print ("type: " + str(dict_Header["CTYPE1"]) + " " + str(dict_Header["CTYPE2"]) + " " + str(dict_Header["CTYPE3"]))
		print ("dimensions: " + str(dict_Header["NAXIS1"]) + " " + str(dict_Header["NAXIS2"]) + " " + str(dict_Header["NAXIS3"]))
		
		fullshape = [dict_Header["NAXIS3"], dict_Header["NAXIS2"], dict_Header["NAXIS1"]]
		
		if len(subcube) == 6:
			np_Cube = f[0].section[subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
			dict_Header["CRPIX1"] -= subcube[0]
			dict_Header["CRPIX2"] -= subcube[2]
			dict_Header["CRPIX3"] -= subcube[4]
			dict_Header["NAXIS1"] = subcube[1] - subcube[0]
			dict_Header["NAXIS2"] = subcube[3] - subcube[2]
			dict_Header["NAXIS3"] = subcube[5] - subcube[4]
		elif not len(subcube):
			np_Cube = f[0].data
		else:
			sys.stderr.write("ERROR: The subcube list must have 6 entries (%i given).\n" % len(subcube))
			raise SystemExit(1)
	# 4 axes:
	elif dict_Header["NAXIS"] == 4:
		if dict_Header["NAXIS4"] != 1:
			print ("type: " + str(dict_Header["CTYPE1"]) + " " + str(dict_Header["CTYPE2"]) + " " + str(dict_Header["CTYPE3"]) + " " + str(dict_Header["CTYPE4"]))
			print ("dimensions: " + str(dict_Header["NAXIS1"]) + " " + str(dict_Header["NAXIS2"]) + " " + str(dict_Header["NAXIS3"]) + " " + str(dict_Header["NAXIS4"]))
			sys.stderr.write("ERROR: The size of the 4th dimension is > 1.\n")
			raise SystemExit(1)
		else:
			fullshape = [dict_Header["NAXIS3"], dict_Header["NAXIS2"], dict_Header["NAXIS1"]]
			
			if len(subcube) == 6:
				np_Cube = f[0].section[0, subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
				dict_Header["CRPIX1"] -= subcube[0]
				dict_Header["CRPIX2"] -= subcube[2]
				dict_Header["CRPIX3"] -= subcube[4]
				dict_Header["NAXIS1"] = subcube[1] - subcube[0]
				dict_Header["NAXIS2"] = subcube[3] - subcube[2]
				dict_Header["NAXIS3"] = subcube[5] - subcube[4]
			elif not len(subcube):
				np_Cube = f[0].section[0]
			else:
				sys.stderr.write("ERROR: The subcube list must have 6 entries (%i given). Ignore 4th axis.\n" % len(subcube))
				raise SystemExit(1)
	# 2 axes:
	elif dict_Header["NAXIS"] == 2:
		sys.stderr.write("WARNING: The input cube has 2 axes, third axis added.\n")
		print ("type: " + str(dict_Header["CTYPE1"]) + " " + str(dict_Header["CTYPE2"]))
		print ("dimensions: " + str(dict_Header["NAXIS1"]) + " " + str(dict_Header["NAXIS2"]))
		
		fullshape = [dict_Header["NAXIS2"], dict_Header["NAXIS1"]]
		
		if len(subcube) == 4:
			np_Cube = array([f[0].section[subcube[2]:subcube[3], subcube[0]:subcube[1]]])
			dict_Header["CRPIX1"] -= subcube[0]
			dict_Header["CRPIX2"] -= subcube[2]
			dict_Header["NAXIS1"] = subcube[1] - subcube[0]
			dict_Header["NAXIS2"] = subcube[3] - subcube[2]
		elif not len(subcube):
			np_Cube = array([f[0].data])
		else:
			sys.stderr.write("ERROR: The subcube list must have 4 entries (%i given).\n" % len(subcube))
			raise SystemExit(1)
	# 1 axis:
	elif dict_Header["NAXIS"] == 1:
		sys.stderr.write("ERROR: The input has 1 axis, this is probably a spectrum instead of a 2D/3D image.\n")
		sys.stderr.write("       Type: " + str(dict_Header["CTYPE1"]) + "\n")
		sys.stderr.write("       Dimensions: " + str(dict_Header["NAXIS1"]) + "\n")
		raise SystemExit(1)
	else:
		sys.stderr.write("ERROR: The file has fewer than 1 or more than 4 dimensions.\n")
		raise SystemExit(1)
	
	f.close()
	
	
	# Check if cube needs to be inverted
	if invertData:
		np_Cube *= -1.0
		print("Inverting data cube to search for negative signals")
	
	
	# check whether the axes are in the expected order:
	#if dict_Header["CTYPE1"][0:2] != "RA" or dict_Header["CTYPE2"][0:3] != "DEC":
	#	sys.stderr.write("WARNING: The dimensions are not in the expected order.\n")
	
	print ("The data cube has been loaded.")
	
	
	if not cubeOnly:
		# Apply weights file if provided:
		if weightsFile:
			# The original data are replaced with the weighted cube!
			# If weighting is being used, the data should be read in again during parameterisation.
			# check whether the weights cube exists:
			if os.path.isfile(weightsFile) == False:
				sys.stderr.write("ERROR: The defined weights cube does not exist.\n")
				sys.stderr.write("       Cannot find: " + weightsFile + "\n")
				raise SystemExit(1)
			else:
				# Scale the input cube with a weights cube
				# load the weights cube and convert it into a 3D array to be applied to the data 3D array
				# (note that the data has been converted into a 3D array above)
				print ("Loading and applying weights cube: " + weightsFile)
				f = fits.open(weightsFile, memmap=False)
				dict_Weights_header = f[0].header
				if dict_Weights_header["NAXIS"] == 3:
					if len(subcube) == 6:
						np_Cube *= f[0].section[subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
					else:
						np_Cube *= f[0].data
				elif dict_Weights_header["NAXIS"] == 4:
					if dict_Weights_header["NAXIS4"] != 1:
						sys.stderr.write("ERROR: The 4th dimension has more than 1 value.\n")
						raise SystemExit(1)
					else:
						sys.stderr.write("WARNING: The weights cube has 4 axes; first axis ignored.\n")
						if len(subcube) == 6: np_Cube *= f[0].section[0, subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
						else: np_Cube *= f[0].section[0]
				elif dict_Weights_header["NAXIS"] == 2:
					sys.stderr.write("WARNING: The weights cube has 2 axes; third axis added.\n")
					if len(subcube) == 6 or len(subcube) == 4: np_Cube *= array([f[0].section[subcube[2]:subcube[3], subcube[0]:subcube[1]]])
					else: np_Cube *= array([f[0].data])
				elif dict_Weights_header["NAXIS"] == 1:
					sys.stderr.write("WARNING: The weights cube has 1 axis; interpreted as third axis; first and second axes added.\n")
					if len(subcube) == 6: np_Cube *= reshape(f[0].section[subcube[4]:subcube[5]], (-1, 1, 1))
					elif not len(subcube): np_Cube *= reshape(f[0].data, (-1, 1, 1))
					else:
						sys.stderr.write("ERROR: The subcube list must have 6 entries (%i given).\n" % len(subcube))
						raise SystemExit(1)
				else:
					sys.stderr.write("ERROR: The weights cube has fewer than 1 or more than 4 dimensions.\n")
					raise SystemExit(1)
				
				f.close()
				print ("Weights cube loaded and applied.")
				
		# Else apply weights function if defined:
		elif weightsFunction:
			# WARNING: I'm not sure if there is a safe way to properly implement multiplication of a data array 
			# WARNING: with a user-specified function in Python without the need for a whitelist, nested loops, 
			# WARNING: or the creation of multiple copies of the cube.
			print ("Evaluating function: %s" % weightsFunction)
			
			# Define whitelist of allowed character sequences:
			whitelist = ["x", "y", "z", "e", "E", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "exp", "log", "sqrt", "square", "power", "absolute", "fabs", "sign"]
			
			# Search for all keywords consisting of consecutive sequences of alphabetical characters:
			# NOTE: Explicit conversion to string is required unless readoptions.py is modified!
			keywordsFound = filter(None, re.split("[^a-zA-Z]+", str(weightsFunction)))
			
			# Check for non-whitelisted sequences:
			for keyword in keywordsFound:
				if keyword not in whitelist:
					sys.stderr.write("ERROR: Unsupported keyword/function found in weights function:\n")
					sys.stderr.write("         %s\n" % weightsFunction)
					sys.stderr.write("       Please check your input.\n")
					raise SystemExit(1)
			
			# ALERT: Why are we not applying the weights function continuously (channel-by-channel)?!?
			# hardcoded number of weights chunks
			Nz = 50
			# check that the number of weights z-chunks is at most equal to the total nr of chans
			Nz = min(Nz, np_Cube.shape[0])
			# calculate the size of each chunk along the Z axis (rounding to integer)
			Lz = int(round(float(np_Cube.shape[0]) / Nz))
			# calculate number of chunks really needed given above rounding
			Nz = int(ceil(float(np_Cube.shape[0]) / Lz))
			print ("Evaluating and applying weights function in %i chunks along the Z axis" % Nz)
			for zz in range(Nz):
				# last chunk may have different length than the others
				if zz == Nz - 1: z, y, x = indices((np_Cube.shape[0] - Lz * zz, np_Cube.shape[1], np_Cube.shape[2]))
				else: z, y, x = indices((Lz, np_Cube.shape[1], np_Cube.shape[2]))
				z += zz * Lz
				try:
					# NOTE: eval() should be safe now as we don't allow for non-whitelisted sequences...
					np_Cube[z,y,x] *= eval(str(weightsFunction))
					# WARNING: There is no check here whether the expression to be evaluated is actually valid,
					#          e.g. SoFiA will crash if the weights function is sqrt(-1). 'try' doesn't catch this!
					#          Even if we set np.seterr(all="raise"), we still run into problems with expressions 
					#          that are valid but not floating-point numbers, e.g. sqrt((1,2)).
				except:
					sys.stderr.write("ERROR: Failed to evaluate weights function:\n")
					sys.stderr.write("         %s\n" % weightsFunction)
					sys.stderr.write("       Please check your input.\n")
					raise SystemExit(1)
			print ("Function-weighted cube created.\n")
		
		
		if doFlag:
			# Apply blanks cube if provided:
			if flagFile:
				# check whether the weights cube exists:
				if os.path.isfile(flagFile) == False:
					sys.stderr.write("ERROR: The defined flag cube does not exist.\n")
					sys.stderr.write("       Cannot find: " + weightsFile + "\n")
					raise SystemExit(1)
				
				# Loading and applying flag file
				print ("Loading and applying flag cube: " + flagFile)
				f = fits.open(flagFile, memmap=False)
				dict_Flag_header = f[0].header
				if dict_Flag_header["NAXIS"] == 3:
					if len(subcube) == 6:
						flags = f[0].section[subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
						np_Cube[isnan(flags)] = nan
					else:
						np_Cube[isnan(f[0].data)] = nan
				elif dict_Flag_header["NAXIS"] == 4:
					if dict_Flag_header["NAXIS4"] != 1:
						sys.stderr.write("ERROR: The 4th dimension has more than 1 value.\n")
						raise SystemExit(1)
					else:
						sys.stderr.write("WARNING: The flag cube has 4 axes; first axis ignored.\n")
						if len(subcube) == 6:
							flags = f[0].section[0, subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
							np_Cube[isnan(flags)] = nan
						else: np_Cube[isnan(f[0].section[0])] = nan
				elif dict_Flag_header["NAXIS"] == 2:
					sys.stderr.write("WARNING: The flag cube has 2 axes; third axis added.\n")
					if len(subcube) == 6 or len(subcube) == 4: 
						flags = f[0].section[subcube[2]:subcube[3], subcube[0]:subcube[1]]
						for channel in range(np_Cube.shape[0]):
							np_Cube[channel][isnan(flags)] = nan
					else: 
						for channel in range(np_Cube.shape[0]):
							np_Cube[channel][isnan(f(0).data)] = nan
				else:
					sys.stderr.write("ERROR: The weights cube has fewer than 1 or more than 4 dimensions.\n")
					raise SystemExit(1)
				
				f.close()
				print ("Flag cube loaded and applied.")
			# if flag regions if provided
			if flagRegions:
				flag(np_Cube, flagRegions)
		
		
		if maskFile:
			# check whether the mask cube exists:
			if not os.path.isfile(maskFile):
				sys.stderr.write("ERROR: The specified mask cube does not exist.\n")
				sys.stderr.write("       Cannot find: " + maskFile + "\n")
				raise SystemExit(1)
			
			else:
				print ("Loading mask cube: " + maskFile)
				g = fits.open(maskFile, memmap=False, do_not_scale_image_data=True)
				dict_Mask_header = g[0].header
				if dict_Mask_header["NAXIS"] == 3:
					if dict_Mask_header["CRVAL1"] != dict_Header["CRVAL1"] or dict_Mask_header["CRVAL2"] != dict_Header["CRVAL2"] or dict_Mask_header["CRVAL3"] != dict_Header["CRVAL3"]:
						err.warning("Input cube and mask may not be on the same WCS grid.")
						err.warning(str(dict_Mask_header["CRVAL1"]) + " " + str(dict_Header["CRVAL1"]) + " " + str(dict_Mask_header["CRVAL2"]) + " " + str(dict_Header["CRVAL2"]) + " " + str(dict_Mask_header["CRVAL3"]) + " " + str(dict_Header["CRVAL3"]))
					if len(subcube) == 6:
						if dict_Mask_header["NAXIS1"] == np_Cube.shape[2] and dict_Mask_header["NAXIS2"] == np_Cube.shape[1] and dict_Mask_header["NAXIS3"] == np_Cube.shape[0]:
							print ("Subcube selection NOT applied to input mask. The full input mask cube matches size of the selected data subcube.")
							mask = g[0].data
						elif dict_Mask_header["NAXIS1"] == fullshape[2] and dict_Mask_header["NAXIS2"] == fullshape[1] and dict_Mask_header["NAXIS3"] == fullshape[0]:
							print ("Subcube selection applied also to input mask. The mask subcube matches size of the selected data subcube.")
							mask = g[0].section[subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
						else:
							sys.stderr.write("ERROR: Neither the full mask nor the subcube of the mask match size of the selected data subcube.\n")
							raise SystemExit(1)
					else: mask = g[0].data
				elif dict_Mask_header["NAXIS"] == 4:
					if dict_Mask_header["CRVAL1"] != dict_Header["CRVAL1"] or dict_Mask_header["CRVAL2"] != dict_Header["CRVAL2"] or dict_Mask_header["CRVAL3"] != dict_Header["CRVAL3"]:
						err.warning("Input cube and mask may not be on the same WCS grid.")
					if dict_Mask_header["NAXIS4"] != 1:
						sys.stderr.write("ERROR: The 4th dimension has more than 1 value.\n")
						raise SystemExit(1)
					elif len(subcube) == 6:
						sys.stderr.write("WARNING: The mask cube has 4 axes; first axis ignored.\n")
						if dict_Mask_header["NAXIS1"] == np_Cube.shape[2] and dict_Mask_header["NAXIS2"] == np_Cube.shape[1] and dict_Mask_header["NAXIS3"] == np_Cube.shape[0]:
							print ("Subcube selection NOT applied to input mask. The full input mask cube matches size of the selected data subcube.")
							mask = g[0].section[0]
						elif dict_Mask_header["NAXIS1"] == fullshape[2] and dict_Mask_header["NAXIS2"] == fullshape[1] and dict_Mask_header["NAXIS3"] == fullshape[0]:
							print ("Subcube selection applied also to input mask. The mask subcube matches size of the selected data subcube.")
							mask = g[0].section[0, subcube[4]:subcube[5], subcube[2]:subcube[3], subcube[0]:subcube[1]]
						else:
							sys.stderr.write("ERROR: Neither the full mask nor the subcube of the mask match size of the selected data subcube.\n")
							raise SystemExit(1)
					else: mask = g[0].section[0]
				elif dict_Mask_header["NAXIS"] == 2:
					if dict_Mask_header["CRVAL1"] != dict_Header["CRVAL1"] or dict_Mask_header["CRVAL2"] != dict_Header["CRVAL2"]:
						err.warning("Input cube and mask may not be on the same WCS grid.")
					sys.stderr.write("WARNING: The mask cube has 2 axes; third axis added.\n")
					if len(subcube) == 6 or len(subcube) == 4:
						if dict_Mask_header["NAXIS1"] == np_Cube.shape[2] and dict_Mask_header["NAXIS2"] == np_Cube.shape[1]:
							print ("Subcube selection NOT applied to input mask. The full input mask cube matches size of the selected data subcube.")
							mask = array([g[0].data])
						elif dict_Mask_header["NAXIS1"] == fullshape[2] and dict_Mask_header["NAXIS2"] == fullshape[1]:
							print ("Subcube selection applied also to input mask. The mask subcube matches size of the selected data subcube.")
							mask = array([g[0].section[subcube[2]:subcube[3], subcube[0]:subcube[1]]])
						else:
							sys.stderr.write("ERROR: Neither the full mask nor the subcube of the mask match size of the selected data subcube.\n")
							raise SystemExit(1)
					else: mask=array([g[0].data])
				elif dict_Mask_header["NAXIS"] == 1:
					sys.stderr.write("WARNING: The mask cube has 1 axis; interpreted as third axis; first and second axes added.\n")
					if dict_Mask_header["CRVAL1"] != dict_Header["CRVAL1"]:
						err.warning("Input cube and mask may not be on the same WCS grid.")
					if len(subcube) == 6:
						if dict_Mask_header["NAXIS1"] == np_Cube.shape[0]:
							print ("Subcube selection NOT applied to input mask. The full input mask cube matches size of the selected data subcube.")
							mask = reshape(g[0].data, (-1, 1, 1))
						elif dict_Mask_header["NAXIS1"] == fullshape[0]:
							print ("Subcube selection applied also to input mask. The mask subcube matches size of the selected data subcube.")
							mask = reshape(g[0].section[subcube[4]:subcube[5]], (-1, 1, 1))
						else:
							sys.stderr.write("ERROR: Neither the full mask nor the subcube of the mask match size of the selected data subcube.\n")
							raise SystemExit(1)
					elif not len(subcube):
						mask = reshape(g[0].data, (-1, 1, 1))
					else:
						sys.stderr.write("ERROR: The subcube list must have 6 entries (%i given).\n" % len(subcube))
						raise SystemExit(1)
				else:
					sys.stderr.write("ERROR: The mask cube has fewer than 1 or more than 4 dimensions.\n")
					raise SystemExit(1)
				#mask[mask > 0] = 1
				g.close()
				print ("Mask cube loaded.")
			
			# In all cases, evaluate import.sources to only keep specific source IDs
			# WARNING: This assumes that source IDs are positive!
			if isinstance(sources, list):
				if sources:
					sys.stdout.write("Only the following sources from the mask are retained:\n  " + str(sources) + "\n")
					for sid in sources:
						mask[mask == sid] *= -1
					mask[mask > 0] *= 0
					mask *= -1
				else:
					sys.stdout.write("All sources from the mask will be retained.\n")
			else:
				sys.stderr.write("\nWARNING: Ignoring parameter 'import.sources'; value is not a valid list.\n\n")
			
			# In all cases, convert mask to Boolean with masked pixels set to 1.
			#mask = (mask > 0).astype(bool)
		else:
			# Create an empty mask if none is provided.
			mask = zeros(np_Cube.shape, dtype=int32)
	
	
	if not cubeOnly:
		return np_Cube, dict_Header, mask, subcube
	else:
		return np_Cube, dict_Header


def flag(cube, regions):
	if not regions: return
	
	print ("Start flagging cube")
	dim = len(cube.shape)
	
	for region in regions:
		if not isinstance(region, list):
			sys.stderr.write("ERROR: Flagging regions must be specified as nested lists\n")
			sys.stderr.write("       of the form [[x1, x2, y1, y2, z1, z2], ...].\n")
			raise SystemExit(1)
		
		try:
			for i in range(0, len(region) // 2):
				if region[2 * i + 1] == "":
					region[2 * i + 1] = cube.shape[dim - i - 1]
			if len(region) == 4:
				cube[0, region[2]:region[3], region[0]:region[1]] = nan
			else:
				cube[region[4]:region[5], region[2]:region[3], region[0]:region[1]] = nan
		except:
			sys.stderr.write("WARNING: Flagging did not succeed. Please check the dimensions of your cube and filters.\n")
	
	print ("Flagging complete")
	
	return
