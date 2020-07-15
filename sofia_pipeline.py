#! /usr/bin/env python


# Import default Python libraries
import sys
import os
from time import time
import numpy as np
import resource

# Import library version numbers
from scipy import __version__ as scipy_version
from astropy import __version__ as astropy_version

# Import SoFiA modules
sys.path.insert(0, os.environ["SOFIA_MODULE_PATH"])
from sofia import functions
from sofia import readoptions
from sofia import import_data
#from sofia import import_data_2
from sofia import sigma_cube
from sofia import pyfind
from sofia import wavelet_finder
from sofia import addrel
from sofia import threshold_filter
from sofia import smooth_cube
from sofia import flagerrors
from sofia import write_filtered_cube
from sofia import writemask
from sofia import writemoment2
#from sofia import writemoment
from sofia import write_catalog
from sofia import linker
from sofia import cubelets
from sofia import parametrisation
from sofia import wcs_coordinates
from sofia import CNHI
from sofia import error as err
from sofia import __version__ as sofia_version

# --------------------------------
# FUNCTION TO MONITOR MEMORY USAGE
# --------------------------------

MEM_FACTOR = 1024.0
if sys.platform == "darwin":
	MEM_FACTOR *= 1024.0

def print_memory_usage(t0):
	err.message("\x1B[36mPeak memory usage: {0:.3f} MB at {1:.3f} s\x1B[0m".format(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / MEM_FACTOR, time() - t0))
	return



# --------------------------------------
# ---- FUNCTION TO CHECK OVERWRITES ----
# --------------------------------------

def checkOverwrite(path):
	if not os.path.exists(path): return
	
	if os.path.isfile(path):
		err.error(
			"Failed to create the output file:\n\n"
			"  " + str(path) + "\n\n"
			"The file already exists. You can do one of the following:\n\n"
			"1) Enable automatic overwrite in the GUI or parameter file\n"
			"2) Change base name and/or output directory in the GUI or\n"
			"   parameter file\n"
			"3) Delete or rename the existing file", fatal=True, frame=True)
	elif os.path.isdir(path) and os.listdir(path):
		err.error(
			"Failed to create the output directory:\n\n"
			"  " + str(path) + "\n\n"
			"The directory already exists and is not empty. You can do one\n"
			"of the following:\n\n"
			"1) Enable automatic overwrite in the GUI or parameter file\n"
			"2) Change base name and/or output directory in the GUI or\n"
			"   parameter file\n"
			"3) Delete or rename the existing directory", fatal=True, frame=True)
	return



# -----------------------------------------------
# ---- Check if parameter file name provided ----
# -----------------------------------------------

if len(sys.argv) != 2:
	err.message("\n\033[1;4mUsage:\033[24m sofia_pipeline.py \033[3m<filename>\033[0m\n\nThe filename of a valid SoFiA parameter file must be specified. Please\nadd the full path if the file is not located in the current directory.\n\n")
	sys.exit(1)



# -----------------------------------------------
# ---- Print some initial status information ----
# -----------------------------------------------

err.print_progress_message("Running the SoFiA pipeline")
err.message(
	"    Using: SoFiA   " + sofia_version + "\n"
	"           Python  " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "." + str(sys.version_info[2]) + "\n"
	"           NumPy   " + np.__version__ + "\n"
	"           SciPy   " + scipy_version + "\n"
	"           Astropy " + astropy_version + "\n")



# --------------------------------
# ---- START OF SoFiA PIPELINE ---
# --------------------------------

t0 = time()



# ---------------------------------
# ---- READ DEFAULT PARAMETERS ----
# ---------------------------------

err.print_progress_message("Reading default parameters", t0)
default_file = os.getenv("SOFIA_PIPELINE_PATH").replace("sofia_pipeline.py", "SoFiA_default_input.txt")
Parameters = readoptions.readPipelineOptions(default_file)



# ------------------------------
# ---- READ USER PARAMETERS ----
# ------------------------------

err.print_progress_message("Reading user parameters", t0)

# This reads in a file with parameters and creates a dictionary:
parameter_file = sys.argv[1]
err.message("Parameters extracted from: " + str(parameter_file))
User_Parameters = readoptions.readPipelineOptions(parameter_file)
if not User_Parameters: err.error("No valid parameter settings found in parameter file.", fatal=True)

# Overwrite default parameters with user parameters (if exist):
for task in iter(User_Parameters):
	if task in Parameters:
		for key in iter(User_Parameters[task]):
			if key in Parameters[task]:
				Parameters[task][key] = User_Parameters[task][key]

# Define the base and output directory name used for output files (defaults to input file 
# name if writeCat.basename is found to be invalid):
outputBase = Parameters["writeCat"]["basename"]
outputDir  = Parameters["writeCat"]["outputDir"]

if outputDir and not os.path.isdir(outputDir):
	err.error("The specified output directory does not exist:\n" + str(outputDir), fatal=True)

if not outputBase or outputBase.isspace() or "/" in outputBase or "\\" in outputBase or outputBase == "." or outputBase == "..":
	outroot = Parameters["import"]["inFile"].split("/")[-1]
	if (outroot.lower()).endswith(".fits") and len(outroot) > 5:
		outroot = outroot[0:-5]
	if (outroot.lower()).endswith(".fits.gz") and len(outroot) > 8:
		outroot = outroot[0:-8]
else:
	outroot = outputBase

if not outputDir or not os.path.isdir(outputDir) or outputDir.isspace():
	outroot = Parameters["import"]["inFile"][0:len(Parameters["import"]["inFile"]) - len(Parameters["import"]["inFile"].split("/")[-1])] + outroot
else:
	if outputDir[-1] != "/": outputDir += "/"
	outroot = outputDir + outroot

if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)

# Transfer scaleNoise arguments to SCfind for the purpose of scaling the noise after each smoothing iteration in the S+C finder
if Parameters["steps"]["doScaleNoise"]:
        for pp in "perSCkernel,method,edgeX,edgeY,edgeZ,scaleX,scaleY,scaleZ,windowSpatial,windowSpectral,gridSpatial,gridSpectral,interpolation".split(','):
                Parameters["SCfind"].update({pp:Parameters["scaleNoise"][pp]})
del(Parameters["scaleNoise"]["perSCkernel"])



# -------------------------------------------
# ---- CHECK FOR FILES TO BE OVERWRITTEN ----
# -------------------------------------------


outputFilteredCube  = str(outroot) + "_filtered.fits"
outputNoiseCube     = str(outroot) + "_noise.fits"
outputSkellamPDF    = str(outroot) + "_rel_skellam.pdf"
outputScatterPDF    = str(outroot) + "_rel_scatter.pdf"
outputContoursPDF   = str(outroot) + "_rel_contour.pdf"
outputDeltaPDF      = str(outroot) + "_rel_skellam-delta.pdf"
outputMaskCube      = str(outroot) + "_mask.fits"
outputMom0Image     = str(outroot) + "_mom0.fits"
outputNrchImage     = str(outroot) + "_nrch.fits"
outputMom1Image     = str(outroot) + "_mom1.fits"
outputCubeletsDir   = str(outroot) + "_cubelets/"
outputCatXml        = str(outroot) + "_cat.xml"
outputCatAscii      = str(outroot) + "_cat.ascii"
outputCatSQL        = str(outroot) + "_cat.sql"
outputCatAsciiDebug = str(outroot) + "_cat.debug.ascii"

if not Parameters["writeCat"]["overwrite"]:
	# Filtered cube
	if Parameters["steps"]["doWriteFilteredCube"] and (Parameters["steps"]["doSmooth"] or Parameters["steps"]["doScaleNoise"] or Parameters["steps"]["doFilterArtefacts"] or Parameters["steps"]["doWavelet"]):
		checkOverwrite(outputFilteredCube)
	
	# Noise cube
	if Parameters["steps"]["doWriteNoiseCube"] and Parameters["steps"]["doScaleNoise"]:
		checkOverwrite(outputNoiseCube)
	
	# Reliability plots
	if Parameters["steps"]["doReliability"] and Parameters["steps"]["doMerge"] and Parameters["reliability"]["makePlot"]:
		checkOverwrite(outputSkellamPDF)
		checkOverwrite(outputScatterPDF)
		checkOverwrite(outputContoursPDF)
		checkOverwrite(outputDeltaPDF)
	
	# Mask
	if Parameters["steps"]["doWriteMask"]:
		checkOverwrite(outputMaskCube)
		
	# Moment maps
	if Parameters["steps"]["doMom0"]:
		checkOverwrite(outputMom0Image)
		checkOverwrite(outputNrchImage)
	if Parameters["steps"]["doMom1"]:
		checkOverwrite(outputMom1Image)
	
	# Cubelet directory
	if Parameters["steps"]["doCubelets"] and Parameters["steps"]["doMerge"]:
		checkOverwrite(outputCubeletsDir)
	
	# Catalogues
	if Parameters["steps"]["doWriteCat"] and Parameters["steps"]["doMerge"] and Parameters["writeCat"]["writeXML"]:
		checkOverwrite(outputCatXml)
	if Parameters["steps"]["doWriteCat"] and Parameters["steps"]["doMerge"] and Parameters["writeCat"]["writeASCII"]:
		checkOverwrite(outputCatAscii)



# --------------------------------------------------------------
# ---- DEFINE LINKER'S OUTPUT AND CHECK RELIBILITY SETTINGS ----
# --------------------------------------------------------------

if Parameters["steps"]["doMerge"]:
	# Define parameters returned by the linker module
	catParNames = ("id", "x_geo", "y_geo", "z_geo", "x", "y", "z", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "n_pix", "snr_min", "snr_max", "snr_sum", "x_p", "y_p", "z_p", "x_n", "y_n", "z_n", "snr_sum_p", "snr_sum_n", "snr_mean", "snr_std", "snr_rms", "w20", "w50", "w20_cfd", "w50_cfd", "n_x", "n_y", "n_chan", "n_los", "fill_frac")
	catParUnits = ("-", "pix", "pix", "chan", "pix", "pix", "chan", "pix", "pix", "pix", "pix", "chan", "chan", "-", "-", "-", "-", "pix", "pix", "chan", "pix", "pix", "chan", "-", "-", "-", "-", "-", "chan", "chan", "chan", "chan", "pix", "pix", "chan", "-", "-")
	catParFormt = ("%10i", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%7i", "%7i", "%7i", "%7i", "%7i", "%7i", "%8i", "%12.3e", "%12.3e", "%12.3e", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%12.3e", "%12.3e", "%12.3e", "%12.3e", "%12.3e", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%7i", "%7i", "%7i", "%7i", "%5.3f")
	
	# --------------------------------------------------------------------------------------
	# ### ALERT: Temporary list of allowed parameters for reliability calculation.
	# ### This is necessary due to a bug in the linker that may produce wrong source parameters
	# ### in some cases. NOTE: This list should be replaced with the original one again once the
	# ### linker has been fixed. Ensure that catParNames_tmp is replaced with catParNames again
	# ### in the for loops below as well!
	# --------------------------------------------------------------------------------------
	catParNames_tmp = ("n_pix", "n_chan", "n_los", "snr_min", "snr_max", "snr_sum", "snr_mean");
	# --------------------------------------------------------------------------------------
	
	# Check that the parameters to be used for the reliability calculation are included in catParNames
	if Parameters["steps"]["doReliability"]:
		for pp in Parameters["reliability"]["parSpace"]:
			if pp not in catParNames_tmp:
				message =  "You requested reliability calculation in the parameter space:\n\n"
				message += "  " + str(Parameters["reliability"]["parSpace"]) + "\n\n"
				message += "However, the parameter " + str(pp) + " is not recognised by SoFiA.\n"
				message += "Allowed parameter names are:\n\n  "
				for i in range(len(catParNames_tmp)):
					message += str(catParNames_tmp[i])
					if i < len(catParNames_tmp) - 1:
						if (i + 1) % 6 == 0: message += ",\n  "
						else: message += ", "
				message += "\n\nPlease use parameter names from the above list and try again."
				# --------------------------------------------------------------------------
				# ALERT: Delete the following again after the linker has been fixed:
				# --------------------------------------------------------------------------
				message += "\n\nNote that there are temporary restrictions in the number of\n"
				message += "parameters available for reliability calculation due to a bug\n"
				message += "in the linker. These restrictions will be lifted again in the\n"
				message += "future once the linker has been fixed. For the time being we\n"
				message += "recommend using the default parameter space of ['snr_mean',\n"
				message += "'snr_sum', 'snr_max'] instead."
				# --------------------------------------------------------------------------
				err.error(message, fatal=True, frame=True)



# ---------------------
# ---- IMPORT DATA ----
# ---------------------

err.print_progress_message("Loading data cube(s)", t0)
kwargs = Parameters["import"].copy()
kwargs.update({"doFlag": Parameters["steps"]["doFlag"], "flagRegions": Parameters["flag"]["regions"], "flagFile": Parameters["flag"]["file"]})
np_Cube, dict_Header, mask, subcube = import_data.read_data(Parameters["steps"]["doSubcube"], **kwargs)
#np_Cube, dict_Header, mask, subcube = import_data_2.import_data(Parameters["steps"]["doSubcube"], **kwargs)
err.message("Data cube(s) loaded.")
if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)


# -------------------------
# ---- PRECONDITIONING ----
# -------------------------

if Parameters["steps"]["doSmooth"] or Parameters["steps"]["doScaleNoise"] or Parameters["steps"]["doWavelet"]:
	err.print_progress_message("Applying input filters", t0)
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)

# ---- SMOOTHING ----
if Parameters["steps"]["doSmooth"]:
	np_Cube = smooth_cube.smooth(np_Cube, **Parameters["smooth"])
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)

# ---- NOISE SCALING ----
if Parameters["steps"]["doScaleNoise"]:
	np_Cube, noise_cube = sigma_cube.sigma_scale(np_Cube, **Parameters["scaleNoise"])
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)

# --- FILTER ARTEFACTS ---
if Parameters["steps"]["doFilterArtefacts"]:
	err.message("Attempting to flag residual continuum emission")
	np_Cube = flagerrors.flaglos(np_Cube, **Parameters["filterArtefacts"])

# --- WAVELET ---
if Parameters["steps"]["doWavelet"]:
        err.message("Running wavelet filter")
        # WARNING: There is a lot of time and memory overhead from transposing the cube forth and back!
        # WARNING: This will need to be addressed in the future.
        np_Cube = np.transpose(np_Cube, axes=[2, 1, 0])
        np_Cube = wavelet_finder.denoise_2d1d(np_Cube, **Parameters["wavelet"])
        np_Cube = np.transpose(np_Cube, axes=[2, 1, 0])
        np_Cube = np_Cube.copy()
        if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)

# --- WRITE FILTERED CUBE ---
if Parameters["steps"]["doWriteFilteredCube"] and (Parameters["steps"]["doSmooth"] or Parameters["steps"]["doScaleNoise"] or Parameters["steps"]["doFilterArtefacts"] or Parameters["steps"]["doWavelet"]):
	err.message("Writing filtered cube")
	write_filtered_cube.writeFilteredCube(np_Cube, dict_Header, Parameters, outputFilteredCube, Parameters["writeCat"]["compress"])
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)

# --- WRITE NOISE CUBE ---
if Parameters["steps"]["doWriteNoiseCube"] and Parameters["steps"]["doScaleNoise"]:
	err.message("Writing noise cube")
	write_filtered_cube.writeFilteredCube(noise_cube, dict_Header, Parameters, outputNoiseCube, Parameters["writeCat"]["compress"])
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)

# --- DELETE NOISE CUBE TO RELEASE MEMORY ---
if Parameters["steps"]["doScaleNoise"]:
	del noise_cube
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)

if Parameters["steps"]["doSmooth"] or Parameters["steps"]["doScaleNoise"] or Parameters["steps"]["doWavelet"]:
	err.message("Input filters applied.")
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)



# ------------------------
# ---- SOURCE FINDING ----
# ------------------------

if Parameters["steps"]["doSCfind"] or Parameters["steps"]["doCNHI"] or Parameters["steps"]["doThreshold"]:
	err.print_progress_message("Running source finder", t0)
	# Turn mask into binary mask
	mask[mask > 0] = 1
	
	# Apply the different filters that each create a mask.
	
	# --- PYFIND (S+C) ---
	if Parameters["steps"]["doSCfind"]:
		err.message("Running S+C filter")
		pyfind.SCfinder_mem(np_Cube, mask, dict_Header, t0, **Parameters["SCfind"])
		if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)
	
	# --- CNHI ---	
	if Parameters["steps"]["doCNHI"]:
		err.message("Running CNHI filter")
		#mask += CNHI.find_sources(np_Cube, mask, **Parameters["CNHI"]) # Fails in Numpy 1.10 or newer due to casting error!
		np.add(mask, CNHI.find_sources(np_Cube, mask, **Parameters["CNHI"]), out=mask, casting="unsafe") # This should work...
		if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)
	
	# --- THRESHOLD ---	
	if Parameters["steps"]["doThreshold"]:
		err.message("Running threshold filter")
		threshold_filter.filter(mask, np_Cube, dict_Header, **Parameters["threshold"])
		if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)
	
	err.message("Source finding complete.")



# --------------------
# ---- POSITIVITY ----
# --------------------

# Check if positivity flag is set; if so, remove negative pixels from mask:
if Parameters["merge"]["positivity"]:
	err.warning(
		"Enabling mask.positivity is dangerous and will render some of SoFiA's\n"
		"most  powerful  algorithms useless,  including mask  optimisation and\n"
		"reliability calculation.  Only use this option if you are fully aware\n"
		"of its risks and consequences!", frame=True)
	mask[np_Cube < 0.0] = 0
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)

# Check whether any pixels are detected
NRdet = (mask > 0).sum()
if not NRdet:
	err.warning("No pixels detected. Exiting pipeline.", fatal=True)
else:
	err.message("{0:,d} out of {1:,d} pixels detected ({2:.4f}%)".format(NRdet, np.array(mask.shape).prod(), 100.0 * float(NRdet) / float(np.array(mask.shape).prod())))



# -------------------------------------
# ---- OUTPUT FOR DEBUGGING (MASK) ----
# -------------------------------------

if Parameters["steps"]["doDebug"] and NRdet:
	err.print_progress_message("Writing binary mask cube for debugging", t0)
	writemask.writeMask(mask, dict_Header, Parameters, outputMaskCube.replace('_mask.fits','_binmask.fits'), Parameters["writeCat"]["compress"], Parameters["writeCat"]["overwrite"])



# -----------------
# ---- MERGING ----
# -----------------

if Parameters["steps"]["doMerge"] and NRdet:
	err.print_progress_message("Merging detections", t0)
	objects = []
	objects, mask = linker.link_objects(np_Cube, objects, mask, Parameters["merge"]["radiusX"], Parameters["merge"]["radiusY"], Parameters["merge"]["radiusZ"], Parameters["merge"]["minSizeX"], Parameters["merge"]["minSizeY"], Parameters["merge"]["minSizeZ"], Parameters["merge"]["maxSizeX"], Parameters["merge"]["maxSizeY"], Parameters["merge"]["maxSizeZ"], Parameters["merge"]["minVoxels"], Parameters["merge"]["maxVoxels"], Parameters["merge"]["minLoS"], Parameters["merge"]["maxLoS"], Parameters["merge"]["minFill"], Parameters["merge"]["maxFill"], Parameters["merge"]["minIntens"], Parameters["merge"]["maxIntens"])
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)
	
	if not objects: err.warning("No objects remain after merging. Exiting pipeline.", fatal=True)
	
	objects = np.array(objects)
	err.message("Merging complete")
	NRdet = len(objects)
	NRdetNeg = (np.array(objects)[:,16] < 0).sum()
	err.message("{0:,d} sources detected: {1:,d} positive and {2:,d} negative.".format(NRdet, NRdet - NRdetNeg, NRdetNeg))
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)
	
	# Set catalogue header
	if "bunit" in dict_Header: dunits = dict_Header["bunit"]
	else: dunits = "-"



# -------------------------------------
# ---- OUTPUT FOR DEBUGGING (MASK) ----
# -------------------------------------

if Parameters["steps"]["doDebug"] and Parameters["steps"]["doMerge"] and NRdet:
	err.print_progress_message("Writing all-source mask cube for debugging", t0)
	writemask.writeMask(mask, dict_Header, Parameters, outputMaskCube.replace('_mask.fits','_allobjmask.fits'), Parameters["writeCat"]["compress"], Parameters["writeCat"]["overwrite"])



# ----------------------------------------------------
# ---- ESTIMATE RELIABILITY FROM NEGATIVE SOURCES ----
# ----------------------------------------------------

if Parameters["steps"]["doReliability"] and Parameters["steps"]["doMerge"] and NRdet:
	if not NRdetNeg:
		err.print_progress_time(t0)
		err.error(
			"You asked SoFiA to calculate the reliability of the detected\n"
			"sources.  Unfortunately, this calculation cannot be done be-\n"
			"cause there are no negative sources  in the catalogue of de-\n"
			"tections.  This could be due  to your source-finding  and/or\n"
			"filtering settings.  Negative sources are strictly necessary\n"
			"to calculate the reliability.\n"
			"You can do one of the following:\n"
			"(1) Switch off the reliability calculation.\n"
			"(2) Modify the source-finding and/or filtering settings in\n"
			"    order to detect negative sources.", fatal=True, frame=True)
	
	# ---- MEASURE GLOBAL RMS AND NORMALISE PARAMETERS----
	err.print_progress_message("Measuring noise to divide flux parameters by global RMS", t0)
	maxNrVox = 1e+6 # maximum number of pixels over which to calculate the global RMS. Sampling below is set accordingly.
	sampleRms = max(1, int((float(np.array(np_Cube.shape).prod()) / maxNrVox)**(1.0 / min(3, len(np_Cube.shape)))))
	globalrms = functions.GetRMS(np_Cube, rmsMode="mad", fluxRange="negative", zoomx=1, zoomy=1, zoomz=1, verbose=True, sample=sampleRms)
	err.print_progress_time(t0)
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)
	
	# normalise flux parameters to global rms
	# (this is done also if weights were applied, in case they are prop. to 1/sigma but not exactly = 1/sigma)
	objects = np.array(objects)
	for scalablePar in ["snr_min","snr_max","snr_sum","snr_sum_p","snr_sum_n","snr_mean","snr_std","snr_rms"]:
		objects[:,catParNames.index(scalablePar)] /= globalrms
	objects = [list(item) for item in list(objects)]
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)
	
	# ---- CALCULATE RELIABILITY ----
	err.print_progress_message("Determining reliability", t0)
	objects, reliable = addrel.EstimateRel(np.array(objects), outroot, catParNames, **Parameters["reliability"])
	err.message("The following {0:d} reliable sources have been detected: {1:}".format(len(reliable),reliable))
	catParNames = tuple(list(catParNames) + ["n_pos",  "n_neg",  "rel"])
	catParUnits = tuple(list(catParUnits) + ["-",      "-",      "-"])
	catParFormt = tuple(list(catParFormt) + ["%12.3e", "%12.3e", "%12.6f"])
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)

elif Parameters["steps"]["doMerge"] and NRdet:
	err.print_progress_time(t0)
	reliable = list(np.array(objects)[np.array(objects)[:,catParNames.index('snr_sum')] > 0,0].astype(int)) # select all positive sources
	err.message("The following {0:d} sources have been detected: {1:}".format(len(reliable),reliable))
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)

else:
	err.print_progress_time(t0)
	reliable = [1,] # if not merging, all detected pixels have ID = 1 and here they are set to be reliable
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)



# -------------------------------------------
# ---- OUTPUT FOR DEBUGGING (CATALOGUES) ----
# -------------------------------------------

if Parameters["steps"]["doDebug"] and Parameters["steps"]["doMerge"] and NRdet:
	err.print_progress_message("Writing separate positive and negative reliability catalogues", t0)

	outputCatAsciiPos = outputCatAscii.replace('_cat.ascii','_rel_poslogcat.ascii')
	objectsPos=np.array(objects)[np.array(objects)[:,catParNames.index('snr_sum')] > 0]
	posPars=['id','x_geo','y_geo','z_geo']+Parameters["reliability"]["parSpace"]+['rel']
	for ppar in posPars: # take log of pos pars
		if ppar[:4]=='snr_':
			objectsPos[:,catParNames.index(ppar)]=np.log10(np.abs(objectsPos[:,catParNames.index(ppar)]))
	objectsPos = [list(item) for item in list(objectsPos)]
	write_catalog.write_catalog_from_array("ASCII", objectsPos, catParNames, catParUnits, catParFormt, posPars, outputCatAsciiPos, Parameters["writeCat"]["compress"], Parameters["writeCat"]["overwrite"], Parameters["parameters"]["getUncertainties"])

	outputCatAsciiNeg = outputCatAscii.replace('_cat.ascii','_rel_neglogcat.ascii')
	objectsNeg=np.array(objects)[np.array(objects)[:,catParNames.index('snr_sum')] < 0]
	negPars=[kk.replace('max','min') for kk in posPars]
	for npar in negPars: # take log of absolute value of neg pars
		if npar[:4]=='snr_':
			objectsNeg[:,catParNames.index(npar)]=np.log10(np.abs(objectsNeg[:,catParNames.index(npar)]))
	objectsNeg = [list(item) for item in list(objectsNeg)]
	write_catalog.write_catalog_from_array("ASCII", objectsNeg, catParNames, catParUnits, catParFormt, negPars, outputCatAsciiNeg, Parameters["writeCat"]["compress"], Parameters["writeCat"]["overwrite"], Parameters["parameters"]["getUncertainties"])



# ------------------------------------------------------
# ---- REMOVE UNNECESSARY PARAMETERS FROM CATALOGUE ----
# ------------------------------------------------------

if Parameters["steps"]["doMerge"] and NRdet:
	objects, catParNames, catParUnits, catParFormt = np.array(objects), list(catParNames), list(catParUnits), list(catParFormt)
	removecols = ["fill_frac", "snr_min", "snr_max", "snr_sum", "x_p", "y_p", "z_p", "x_n", "y_n", "z_n", "snr_sum_p", "snr_sum_n", "snr_mean", "snr_std", "snr_rms", "w20", "w50", "w20_cfd", "w50_cfd", "n_pos", "n_neg", "n_x", "n_y"]
	
	for remcol in removecols:
		if remcol in catParNames:
			index = catParNames.index(remcol)
			del(catParNames[index])
			del(catParUnits[index])
			del(catParFormt[index])
			objects = np.delete(objects, [index], axis=1)
	
	objects, catParNames, catParUnits, catParFormt = [list(item) for item in list(objects)], tuple(catParNames), tuple(catParUnits), tuple(catParFormt)
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)



# --------------------------------------------------
# ---- REMOVE NON RELIABLE AND NEGATIVE SOURCES ----
# --------------------------------------------------

if Parameters["steps"]["doMerge"] and NRdet:
	err.print_progress_message("Removing unreliable sources", t0)
	
	# Make sure that reliable is sorted
	relList = list(reliable)
	relList.sort()
	reliable = np.array(relList)
	
	# Remove non-reliable sources in the objects array
	relObjects = []
	for rr in reliable:
		relObjects.append([len(relObjects) + 1] + list(objects[rr - 1]))
	relObjects = np.array(relObjects)
	objects = relObjects
	
	tmpCatParNames = list(catParNames);
	tmpCatParNames.insert(1, "id_old");
	catParNames = tuple(tmpCatParNames);
	
	tmpCatParFormt = list(catParFormt);
	tmpCatParFormt.insert(1, "%10i");
	catParFormt= tuple(tmpCatParFormt);
	
	tmpCatParUnits = list(catParUnits);
	tmpCatParUnits.insert(1, "-");
	catParUnits= tuple(tmpCatParUnits);
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)
	
	# In the mask file
	mask *= -1
	index = 1
	catParNames = np.array(catParNames)
	for rr in reliable:
		objrr = objects[objects[:,1] == rr][0]
		Xmin  = int(objrr[catParNames == "x_min"])
		Ymin  = int(objrr[catParNames == "y_min"])
		Zmin  = int(objrr[catParNames == "z_min"])
		Xmax  = int(objrr[catParNames == "x_max"])
		Ymax  = int(objrr[catParNames == "y_max"])
		Zmax  = int(objrr[catParNames == "z_max"])
		mask[Zmin:Zmax+1, Ymin:Ymax+1, Xmin:Xmax+1][mask[Zmin:Zmax+1, Ymin:Ymax+1, Xmin:Xmax+1] == -rr] = index
		index += 1
	mask[mask < 0] = 0
	catParNames = tuple(catParNames)
	
	newRel = []
	for i in range(0, len(relObjects)):
		newRel.append(i + 1)
	reliable = np.array(newRel)
	NRdet = objects.shape[0]
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)



# --------------------------------------
# Terminate if no reliable sources found
# --------------------------------------

if not NRdet:
	err.warning("No sources detected. Exiting pipeline.", fatal=True)



# -------------------------------------------------------------------------------
# ---- RELOAD ORIGINAL DATA CUBE FOR PARAMETERISATION IF IT HAS BEEN CHANGED ----
# -------------------------------------------------------------------------------

if Parameters["steps"]["doSmooth"] or Parameters["steps"]["doScaleNoise"] or Parameters["import"]["weightsFile"] or Parameters["import"]["weightsFunction"]:
	err.message("Reloading data cube for parameterisation")
	del np_Cube, dict_Header
	kwargs = Parameters["import"].copy()
	kwargs.update({"cubeOnly":True})
	np_Cube, dict_Header = import_data.read_data(Parameters["steps"]["doSubcube"], **kwargs)
	#np_Cube, dict_Header = import_data_2.import_data(Parameters["steps"]["doSubcube"], **kwargs)
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)



# ----------------------------------------
# ---- OUTPUT FOR DEBUGGING (MOMENTS) ----
# ----------------------------------------

if Parameters["steps"]["doDebug"]:
	#err.print_progress_message("Writing pre-optimisation mask and moment maps for debugging", t0)
	debug = 1
	#writemask.writeMask(mask, dict_Header, Parameters, "%s_mask.debug_rel.fits" % outroot, Parameters["writeCat"]["compress"])
	#mom0_Image = writemoment.writeMoment0(np_Cube, mask, outroot, debug, dict_Header, Parameters["writeCat"]["compress"])
	#writemoment.writeMoment1(np_Cube, mask, outroot, debug, dict_Header, mom0_Image, Parameters["writeCat"]["compress"])



# ----------------------
# ---- PARAMETERISE ----
# ----------------------

if Parameters["steps"]["doParameterise"]:
	if not Parameters["steps"]["doMerge"]:
		catParNames, catParUnits, catParFormt, objects, dunits = parametrisation.parameters_from_mask(dict_Header, mask)
	err.print_progress_message("Parameterising sources", t0)
	
	# Print warning message about statistical uncertainties
	if Parameters["parameters"]["getUncertainties"]:
		err.warning(
			"   You have requested statistical uncertainties for\n"
			"several source parameters. Please be aware that the\n"
			"calculation of statistical uncertainties depends on\n"
			"a number of assumptions that may not be met. Hence,\n"
			"the resulting numbers  may not be representative of\n"
			"the true uncertainties of those parameters, in par-\n"
			"ticular in the presence of systematic errors.", frame=True)
	
	if Parameters["parameters"]["dilateMask"]: mask, objects = parametrisation.dilate(np_Cube, mask, objects, catParNames, Parameters)
	np_Cube, mask, objects, catParNames, catParFormt, catParUnits = parametrisation.parametrise(np_Cube, mask, objects, catParNames, catParFormt, catParUnits, Parameters, dunits)
	catParNames = tuple(catParNames)
	catParUnits = tuple(catParUnits)
	catParFormt = tuple(catParFormt)
	
	err.message("Parameterisation complete.")
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)



# -----------------------------------------
# ---- REMEMBER IF OBJECT ARRAY EXISTS ----
# -----------------------------------------

object_array_exists = "objects" in locals()



# --------------------
# ---- WRITE MASK ----
# --------------------

if Parameters["steps"]["doWriteMask"]:
	err.print_progress_message("Writing mask cube", t0)
	writemask.writeMask(mask, dict_Header, Parameters, outputMaskCube, Parameters["writeCat"]["compress"], Parameters["writeCat"]["overwrite"])
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)



# ------------------------
# ---- STORE CUBELETS ----
# ------------------------

if Parameters["steps"]["doCubelets"] and object_array_exists:
	err.print_progress_message("Writing cubelets", t0)
	objects = np.array(objects)
	cathead = np.array(catParNames)
	cubelets.writeSubcube(np_Cube, dict_Header, mask, objects, cathead, outroot, outputCubeletsDir, Parameters["writeCat"]["compress"], Parameters["writeCat"]["overwrite"])
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)



# ----------------------------
# ---- MAKE MOM0 and MOM1 ----
# ----------------------------

if (Parameters["steps"]["doMom0"] or Parameters["steps"]["doMom1"]):
	err.print_progress_message("Writing moment maps", t0)
	debug = 0
	write_mom = [Parameters["steps"]["doMom0"], Parameters["steps"]["doMom1"], False]
	writemoment2.writeMoments(np_Cube, mask, outroot, debug, dict_Header, Parameters["writeCat"]["compress"], write_mom[0], write_mom[1], Parameters["writeCat"]["overwrite"])
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)
	#writemoment.writeMoments(np_Cube, mask, outroot, debug, dict_Header, Parameters["writeCat"]["compress"], write_mom, Parameters["writeCat"]["overwrite"])
	
	# WARNING: This will regrid and hence alter the data cube!
	#          Read the original data cube again if needed for further
	#          processing beyond this point:
	#err.print_progress_message("Reloading original data cube", t0)
	##np_Cube, dict_Header, mask, subcube = import_data.read_data(Parameters["steps"]["doSubcube"], **Parameters["import"])
	#np_Cube, dict_Header, mask, subcube = import_data_2.import_data(Parameters["steps"]["doSubcube"], **Parameters["import"])



# ----------------------------------------------------
# ---- CORRECT COORDINATES IF WORKING ON SUBCUBES ----
# ----------------------------------------------------

if len(subcube) and object_array_exists:
	err.print_progress_message("Correcting parameters for sub-cube offset", t0)
	# List of parameters to correct for X, Y and Z offset
	corrX = ["x_geo", "x", "x_min", "x_max"]
	corrY = ["y_geo", "y", "y_min", "y_max"]
	corrZ = ["z_geo", "z", "z_min", "z_max", "bf_z"]
	
	if subcube[0]:
		for pp in corrX:
			if pp in catParNames: objects[:,list(catParNames).index(pp)] += subcube[0]
	if subcube[2]:
		for pp in corrY:
			if pp in catParNames: objects[:,list(catParNames).index(pp)] += subcube[2]
	if subcube[4]:
		for pp in corrZ:
			if pp in catParNames: objects[:,list(catParNames).index(pp)] += subcube[4]



# ---------------------------------------------------
# ---- APPEND PARAMETER VALUES IN PHYSICAL UNITS ----
# ---------------------------------------------------

if Parameters["steps"]["doWriteCat"] and object_array_exists:
	err.print_progress_message("Adding WCS position to catalogue", t0)
	objects, catParNames, catParFormt, catParUnits = wcs_coordinates.add_wcs_coordinates(objects, catParNames, catParFormt, catParUnits, Parameters)



# --------------------------
# ---- STORE CATALOGUES ----
# --------------------------

if Parameters["steps"]["doWriteCat"] and object_array_exists:
	err.print_progress_message("Writing output catalogue", t0)
	
	if "rms" in catParNames:
		catParFormt=list(catParFormt)
		catParFormt[list(catParNames).index("rms")] = "%12.4e"
		catParFormt=tuple(catParFormt)
	
	if Parameters["writeCat"]["writeXML"]:
		write_catalog.write_catalog_from_array("XML", objects, catParNames, catParUnits, catParFormt, Parameters["writeCat"]["parameters"], outputCatXml, Parameters["writeCat"]["compress"], Parameters["writeCat"]["overwrite"], Parameters["parameters"]["getUncertainties"])
	
	if Parameters["writeCat"]["writeASCII"]:
		write_catalog.write_catalog_from_array("ASCII", objects, catParNames, catParUnits, catParFormt, Parameters["writeCat"]["parameters"], outputCatAscii, Parameters["writeCat"]["compress"], Parameters["writeCat"]["overwrite"], Parameters["parameters"]["getUncertainties"])
	
	if Parameters["writeCat"]["writeSQL"]:
		write_catalog.write_catalog_from_array("SQL", objects, catParNames, catParUnits, catParFormt, Parameters["writeCat"]["parameters"], outputCatSQL, Parameters["writeCat"]["compress"], Parameters["writeCat"]["overwrite"], Parameters["parameters"]["getUncertainties"])
	if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)



err.print_progress_message("Pipeline finished", t0)
if Parameters["pipeline"]["trackMemory"]: print_memory_usage(t0)
