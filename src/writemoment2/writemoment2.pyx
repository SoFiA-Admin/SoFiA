#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
import astropy.io.fits as pyfits
import os
import numpy as np
cimport numpy as np
import scipy.constants
from scipy import interpolate
from libc.math cimport isnan
from sofia import functions as func
from sofia import error as err
from sofia import __version_full__ as sofia_version_full
from sofia import __astropy_arg_overwrite__



def regridMaskedChannels(datacube,maskcube,header):
	maskcubeFlt = maskcube.astype("float")
	maskcubeFlt[maskcube > 1] = 1.0

	err.message("Regridding...")
	z = (np.arange(1.0, header["naxis3"] + 1) - header["CRPIX3"]) * header["CDELT3"] + header["CRVAL3"]

	if func.check_header_keywords(func.KEYWORDS_VELO, header["CTYPE3"]):
		pixscale = (1.0 - header["CRVAL3"] / scipy.constants.c) / (1.0 - z / scipy.constants.c)
	elif func.check_header_keywords(func.KEYWORDS_FREQ, header["CTYPE3"]):
		pixscale = header["CRVAL3"] / z
	else:
		err.warning("Cannot convert 3rd axis coordinates to frequency.\nIgnoring the effect of CELLSCAL = 1/F.")
		pixscale = np.ones((header["naxis3"]))

	x0 = header["crpix1"] - 1
	y0 = header["crpix2"] - 1
	xs = np.arange(datacube.shape[2], dtype=float) - x0
	ys = np.arange(datacube.shape[1], dtype=float) - y0

	regridThreshold=[]
	for zz in range(datacube.shape[0]):
		regrid_channel = interpolate.RectBivariateSpline(ys * pixscale[zz], xs * pixscale[zz], datacube[zz])
		datacube[zz] = regrid_channel(ys, xs)
		regrid_channel_mask = interpolate.RectBivariateSpline(ys * pixscale[zz], xs * pixscale[zz], maskcubeFlt[zz])
		maskcubeFlt[zz] = regrid_channel_mask(ys, xs)
		regridThreshold.append(maskcubeFlt[zz].min())
		maskcubeFlt[zz] = np.abs(maskcubeFlt[zz])

	regridThreshold = abs(min(regridThreshold))
	datacube[maskcubeFlt <= regridThreshold] = 0.0
	del maskcubeFlt

	return datacube


def writeMoments(datacube, maskcube, filename, debug, header, compress, domom0, domom1, flagOverwrite):
	# ---------------------------
	# Number of detected channels
	# ---------------------------
	nrdetchan = (maskcube > 0).sum(axis=0)
	if nrdetchan.max() < 65535:
		nrdetchan = nrdetchan.astype("int16")
	else:
		nrdetchan = nrdetchan.astype("int32")

	hdu = pyfits.PrimaryHDU(data=nrdetchan, header=header)
	hdu.header["BUNIT"] = "Nchan"
	hdu.header["DATAMIN"] = nrdetchan.min()
	hdu.header["DATAMAX"] = nrdetchan.max()
	hdu.header["ORIGIN"] = sofia_version_full
	func.delete_header(hdu.header, "CRPIX3")
	func.delete_header(hdu.header, "CRVAL3")
	func.delete_header(hdu.header, "CDELT3")
	func.delete_header(hdu.header, "CTYPE3")

	name = "%s_nrch.fits" % filename
	if compress: name += ".gz"

	# Check for overwrite flag
	if func.check_overwrite(name, flagOverwrite):
		hdu.writeto(name, output_verify="warn", **__astropy_arg_overwrite__)

	# WARNING: The generation of moment maps will mask the copy of the data cube held
	#          in memory by SoFiA. If you wish to use the original data cube after
	#          this point, please reload it first!
	datacube[maskcube == 0] = 0

	if "CELLSCAL" in header and header["CELLSCAL"] == "1/F":
		err.warning(
			"CELLSCAL keyword with value of 1/F found.\n"
			"Will regrid masked cube before making moment images.")
		datacube = regridMaskedChannels(datacube, maskcube, header)

	datacube = np.array(datacube, dtype=np.single)

	# Calculate moment 0
	if domom0 or domom1: m0 = mom0(datacube)
	# NOTE: Shouldn't the following be a lot faster? This uses built-in NumPy routines
	#       which are written in plain C and probably as fast as it can get.
	# NOTE: I ran a few tests which suggest that calling the np.nansum() function is
	#       almost three times as fast as the mom0() function on the test data cube!
	#       Results: 0.013 seconds with np.nansum()
	#                0.033 seconds with mom0()
	# NOTE: NumPy uses way too much memory, so unfortunately cannot be used!
	#if domom0 or domom1:
	#	m0 = np.nansum(datacube, axis=0)

	# --------------
	# Moment 0 image
	# --------------
	if domom0:
		if "CTYPE3" in header:
			err.message("Writing moment-0") # in units of header["BUNIT"] * header["CDELT3"]

			# Velocity
			if "vopt" in header["CTYPE3"].lower() or "vrad" in header["CTYPE3"].lower() or "velo" in header["CTYPE3"].lower() or "felo" in header["CTYPE3"].lower():
				if not "CUNIT3" in header or header["CUNIT3"].lower() == "m/s":
					# Converting (assumed) m/s to km/s
					dkms = abs(header["CDELT3"]) / 1e+3
					scalemom12 = 1.0 / 1e+3
					bunitExt = ".km/s"
				elif header["CUNIT3"].lower() == "km/s":
					# Working in km/s
					dkms = abs(header["CDELT3"])
					scalemom12 = 1.0
					bunitExt = ".km/s"
				else:
					# Working with whatever units the cube has
					dkms = abs(header["CDELT3"])
					scalemom12 = 1.0
					bunitExt = "." + header["CUNIT3"]

			# Frequency
			elif "freq" in header["CTYPE3"].lower():
				if not "CUNIT3" in header or header["CUNIT3"].lower() == "hz":
					# Using (or assuming) Hz
					dkms = abs(header["CDELT3"])
					scalemom12 = 1.0
					bunitExt = ".Hz"
				elif header["CUNIT3"].lower() == "khz":
					# Converting kHz to Hz
					dkms = abs(header["CDELT3"]) * 1e+3
					scalemom12 = 1e+3
					bunitExt = ".Hz"
				else:
					# Working with whatever frequency units the cube has
					dkms = abs(header["CDELT3"])
					scalemom12 = 1.0
					bunitExt = "." + header["CUNIT3"]

			hdu = pyfits.PrimaryHDU(data=m0*dkms, header=header)
			if "BUNIT" in hdu.header: hdu.header["BUNIT"] += bunitExt
			hdu.header["DATAMIN"] = (m0 * dkms).min()
			hdu.header["DATAMAX"] = (m0 * dkms).max()
			hdu.header["ORIGIN"] = sofia_version_full
			hdu.header["CELLSCAL"] = "constant"
			func.delete_header(hdu.header, "CRPIX3")
			func.delete_header(hdu.header, "CRVAL3")
			func.delete_header(hdu.header, "CDELT3")
			func.delete_header(hdu.header, "CTYPE3")
		else:
			hdu = pyfits.PrimaryHDU(data=m0, header=header)
			hdu.header["DATAMIN"] = m0.min()
			hdu.header["DATAMAX"] = m0.max()
			hdu.header["ORIGIN"] = sofia_version_full
			domom1 = False

		if debug:
			hdu.writeto("%s_mom0.debug.fits" % filename, output_verify="warn", **__astropy_arg_overwrite__)
		else:
			name = "%s_mom0.fits" % filename
			if compress: name += ".gz"

			# Check for overwrite flag
			if func.check_overwrite(name, flagOverwrite):
				hdu.writeto(name, output_verify="warn", **__astropy_arg_overwrite__)

	# --------------
	# Moment 1 image
	# --------------
	if domom1:
		err.message("Writing moment-1")

		# Calculate moment 1
		m1 = mom1(datacube, m0, header["CRPIX3"], header["CRVAL3"], header["CDELT3"])
		# NOTE: Again, NumPy is a lot faster than the user-defined mom1() function!
		#       Tests on the SoFiA test data cube yield:
		#       Results: 0.039 seconds with mom1()
		#                0.023 seconds with NumPy
		# NOTE: NumPy uses way too much memory, so unfortunately cannot be used!
		#tmp = ((np.arange(datacube.shape[0]) + 1.0 - header["CRPIX3"]) * header["CDELT3"] + header["CRVAL3"]).reshape((datacube.shape[0], 1, 1))
		#with np.errstate(invalid="ignore"):
		#	m1 = np.divide(np.nansum(datacube * tmp, axis=0), m0)


		# Velocity
		if "vopt" in header["CTYPE3"].lower() or "vrad" in header["CTYPE3"].lower() or "velo" in header["CTYPE3"].lower() or "felo" in header["CTYPE3"].lower():
			if not "CUNIT3" in header:
				m1 /= 1e+3 # Assuming m/s
				bunitExt = "km/s"
			elif header["CUNIT3"].lower() == "km/s":
				bunitExt = "km/s"
			else:
				bunitExt = header["CUNIT3"]

		# Frequency
		elif "freq" in header["CTYPE3"].lower():
			if not "CUNIT3" in header or header["CUNIT3"].lower() == "hz":
				bunitExt = "Hz"
			else:
				bunitExt = header["CUNIT3"]
			dkms = 1.0 # No scaling, avoids crashing

		hdu = pyfits.PrimaryHDU(data=m1, header=header)
		hdu.header["BUNIT"] = bunitExt
		hdu.header["DATAMIN"] = np.nanmin(m1)
		hdu.header["DATAMAX"] = np.nanmax(m1)
		hdu.header["ORIGIN"] = sofia_version_full
		hdu.header["CELLSCAL"] = "constant"
		func.delete_header(hdu.header, "CRPIX3")
		func.delete_header(hdu.header, "CRVAL3")
		func.delete_header(hdu.header, "CDELT3")
		func.delete_header(hdu.header, "CTYPE3")

		if debug:
			hdu.writeto("%s_mom1.debug.fits" % filename, output_verify="warn", **__astropy_arg_overwrite__)
		else:
			name = "%s_mom1.fits" % filename
			if compress: name += ".gz"

			# Check for overwrite flag
			if func.check_overwrite(name, flagOverwrite):
				hdu.writeto(name, output_verify="warn", **__astropy_arg_overwrite__)


def mom0(cube1):
	cdef:
		int i, j, k
		double[:,:] mom0 = np.zeros((cube1.shape[1], cube1.shape[2]))
		float[:,:,:] cube = cube1

	for j in range(cube.shape[1]):
		for k in range(cube.shape[2]):
			for i in range(cube.shape[0]):
				if not isnan(cube[i, j, k]):
					mom0[j, k] += cube[i, j, k]

	return np.array(mom0)


def mom1(cube1, cube2, int cpx, float cval, float cdelt):
	cdef:
		int i, j, k
		double sum
		double[:,:] mom1 = np.zeros((cube1.shape[1], cube1.shape[2]))
		float[:,:,:] cube = cube1
		double[:,:] mom0 = cube2

	for j in range(cube.shape[1]):
		for k in range(cube.shape[2]):
			sum = 0
			for i in range(cube.shape[0]):
				if not isnan(cube[i, j, k]):
					sum += cube[i, j, k] * ((i + 1 - cpx) * cdelt + cval)

			if mom0[j, k] != 0 and not isnan(mom0[j, k]):
				mom1[j, k] = sum / mom0[j, k]
			else:
				mom1[j, k] = np.nan

	return np.array(mom1)
