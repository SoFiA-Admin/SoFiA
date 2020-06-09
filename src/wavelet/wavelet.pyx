#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=False

# Compile time constants
DEF ZERO_BOUNDS = False
DEF WRAP_BOUNDS = True
# Unless your datacube is CRAZY BIG, 2 threads is enough
#DEF NTHREADS = 2

cimport numpy as np
import numpy as np
#from cython.parallel import prange

# Using Python bool!
from cpython cimport bool

cdef inline int index_reflect(int index, int size) nogil:
    while index < 0 or index >= size:

        if index < 0: index = index * -1
        if index >= size: index = 2 * (size - 1) - index

    return index

#cdef inline bool in_bounds(int index, int size) nogil:
#    return index >= 0 and index < size

cdef void convolve_with_stepsize(float[:] source, float[:] target, float[:] kernel, int stepsize = 1):
    cdef:
        int size = source.shape[0]
        int kernelsize = kernel.shape[0]
        int index, i, j
        double a

    for i in range(size):
        index = i - kernelsize // 2 * stepsize
        a = 0.0
        for j in range(kernelsize):
            IF ZERO_BOUNDS:
                if index >= 0 and index < size:
                    a += <double>source[index] * <double>kernel[j]
            ELIF WRAP_BOUNDS:
                a += <double>source[index % size] * <double>kernel[j]
            ELSE:
                a += <double>source[index_reflect(index, size)] * <double>kernel[j]
            index += stepsize
        target[i] = <float>a

cdef void convolve_x_with_stepsize(float[:,:,:] source, float[:,:,:] target, float[:] kernel, int stepsize = 1) nogil:
    cdef:
        int zsize = source.shape[0]
        int ysize = source.shape[1]
        int xsize = source.shape[2]
        int kernelsize = kernel.shape[0]
        int index, z, y, x, j

    #for z in prange(zsize, num_threads=NTHREADS):
    for z in range(zsize):
        for y in range(ysize):
            for x in range(xsize):
                index = x - kernelsize // 2 * stepsize
                target[z,y,x] = 0.0
                for j in range(kernelsize):
                    IF ZERO_BOUNDS:
                        if index >= 0 and index < xsize:
                            target[z,y,x] += source[z, y, index] * kernel[j]
                    ELIF WRAP_BOUNDS:
                        target[z,y,x] += source[z, y, index % xsize] * kernel[j]
                    ELSE:
                        target[z,y,x] += source[z, y, index_reflect(index, xsize)] * kernel[j]
                    index = index + stepsize


cdef void convolve_y_with_stepsize(float[:,:,:] source, float[:,:,:] target, float[:] kernel, int stepsize = 1) nogil:
    cdef:
        int zsize = source.shape[0]
        int ysize = source.shape[1]
        int xsize = source.shape[2]
        int kernelsize = kernel.shape[0]
        int index, z, y, x, j

    #for z in prange(zsize, num_threads=NTHREADS):
    for z in range(zsize):
        for y in range(ysize):
            for x in range(xsize):
                index = y - kernelsize // 2 * stepsize
                target[z,y,x] = 0.0
                for j in range(kernelsize):
                    IF ZERO_BOUNDS:
                        if index >= 0 and index < ysize:
                            target[z,y,x] += source[z, index, x] * kernel[j]
                    ELIF WRAP_BOUNDS:
                        target[z,y,x] += source[z, index % ysize, x] * kernel[j]
                    ELSE:
                        target[z,y,x] += source[z, index_reflect(index,ysize), x] * kernel[j]
                    index = index + stepsize


cdef void convolve_z_with_stepsize(float[:,:,:] source, float[:,:,:] target, float[:] kernel, int stepsize = 1) nogil:
    cdef:
        int zsize = source.shape[0]
        int ysize = source.shape[1]
        int xsize = source.shape[2]
        int kernelsize = kernel.shape[0]
        int index, z, y, x, j

    #for z in prange(zsize, num_threads=NTHREADS):
    for z in range(zsize):
        for y in range(ysize):
            for x in range(xsize):
                index = z - kernelsize // 2 * stepsize
                target[z,y,x] = 0.0
                for j in range(kernelsize):
                    IF ZERO_BOUNDS:
                        if index >= 0 and index < zsize:
                            target[z,y,x] += source[index, y, x] * kernel[j]
                    ELIF WRAP_BOUNDS:
                        target[z,y,x] += source[index % zsize, y, x] * kernel[j]
                    ELSE:
                        target[z,y,x] += source[index_reflect(index,zsize), y, x] * kernel[j]
                    index = index + stepsize

cdef inline void inplace_diff(float[:,:,:] a, float[:,:,:] b):
    """
    Performs the operation

    a = a - b

    in place for 3D memoryviews.
    """
    cdef:
        int i,j,k

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                a[i,j,k] = a[i,j,k] - b[i,j,k]


# QUESTION: The function currently returns -1 for x = 0. Is this intended?
cdef inline float sign(float x):
    if x > 0.0:
        return 1.0
    else:
        return -1.0


cdef class WaveletDecomposition2D1D:
    """
    Base class for 2D-1D wavelet decomposition after Starck et al. 2009 and Flöer & Winkel 2012.
    """

    cdef:
        np.ndarray _work, _data
        int _xy_scales, _z_scales
        np.ndarray _xy_mother_function, _z_mother_function


    def __init__(self, data, xy_scales=-1, z_scales=-1):
        """
        Initialize the decomposition object.

        Parameters
        ----------

        data : 3D numpy array
            The data to decompose. The data are converted to float, i.e. single accuracy, values.

        xy_scales, z_scales : integer, optional
            The spatial and spectral scales to use for data decomposition.
            If either is less than 0 or not given, the scales are determined from the data, e.g. z_scales = ceil(log_2(data.shape[0])).
            If either scale is 0, no decomposition is performed for the corresponding axes.
        """

        self._data = data.astype(np.single)
        self._work = np.zeros((3,) + data.shape , dtype = np.single)

        self.xy_scales = xy_scales
        self.z_scales = z_scales


    property work:
        """
        ndarray containing the working space of the transform
        """
        def __get__(self):
            return self._work


    property data:
        """
        ndarray containing the original data to be reonstructed
        """
        def __get__(self):
            return self._data
        def __set__(self, value):
            self._data = np.array(value, dtype=np.single)


    property xy_mother_function:
        """
        ndarray containing the wavelet mother function
        for the spatial part of the transform
        """
        def __get__(self):
            return self._xy_mother_function
        def __set__(self, value):
            self._xy_mother_function = np.array(value, dtype=np.single)


    property z_mother_function:
        """
        narray containing the wavelet mother function
        for the spectral part of the transform
        """
        def __get__(self):
            return self._z_mother_function
        def __set__(self, value):
            self._z_mother_function = np.array(value, dtype=np.single)


    property xy_scales:
        """
        the number of spatial scales to use during the reconstruction
        """
        def __get__(self):
            return self._xy_scales
        def __set__(self, value):
            if value < 0:
                self._xy_scales = np.floor(np.log(max(self.data.shape[1], self.data.shape[2])) / np.log(2.0)).astype(np.intc)
            else:
                self._xy_scales = np.floor(value).astype(np.intc)


    property z_scales:
        """
        The number of spectral scales to use during the reconstruction
        """
        def __get__(self):
            return self._z_scales
        def __set__(self, value):
            if value < 0:
                self._z_scales = np.floor(np.log(self.data.shape[0]) / np.log(2.0)).astype(np.intc)
            else:
                self._z_scales = np.floor(value).astype(np.intc)


    def decompose(self):
        """
        Perform the wavelet decomposition. For each combination of spatial and spectral scale,
        the coefficients are calculated and the handle_coefficients member function is called.
        """

        cdef:
            float[:,:,:] data = self._data
            float[:,:,:,:] work = self._work

            int xy_scale_factor, z_scale_factor

            int fine_xy, fine_z, coarse_xy, coarse_z
            int i, j, k

        xy_scale_factor = 1

        coarse_xy = 0
        fine_xy = 1

        self.init_work()

        for xy_scale from 0 <= xy_scale <= self.xy_scales:

            if xy_scale < self.xy_scales:
                # Convolve the coarse data with the necessary kernel and step sizes
                convolve_x_with_stepsize(work[coarse_xy], work[2], self._xy_mother_function, xy_scale_factor)
                convolve_y_with_stepsize(work[2], work[fine_xy], self._xy_mother_function, xy_scale_factor)
                # Now, the previous coarse scale is in coarse_xy and the current coarse scale is in fine_xy

                # Create the wavelet coefficients in coarse_xy by subtracting
                # the current smooth scale contained in fine_xy
                inplace_diff(work[coarse_xy], work[fine_xy])
                # Now, the wavelet coefficients are in coarse_xy

                # Exchange the indices
                coarse_xy = (coarse_xy + 1) % 2
                fine_xy = (fine_xy + 1) % 2
                # Now the current coarse scale is in coarse_xy and the wavelet coefficients are in fine_xy
            else:
                # In the last iteration, the "wavelet coefficients" to be analyzed by
                # the spectral decomposition is just the maximally smoothed spatial scale
                fine_xy = coarse_xy

            # Initialize the scale factor for spectral decomposition
            # and set the coarse_z index to point to the xy wavelet coefficients
            # and the fine_z index to point to the third work array
            z_scale_factor = 1
            coarse_z = fine_xy
            fine_z = 2

            for z_scale from 0 <= z_scale < self.z_scales:

                # Convolve the coarse_z values with the appropriate kernel and step size
                convolve_z_with_stepsize(work[coarse_z], work[fine_z], self._z_mother_function, z_scale_factor)
                # Now, the previous coarse version is in coarse_z and the current coarse version is in fine_z

                # Create the wavelet coefficients in coarse_z
                inplace_diff(work[coarse_z], work[fine_z])
                # Now the wavelet coefficients are in coarse_z

                # Exchange the indices and take care not to overwrite coarse_xy
                if z_scale % 2 == 0:
                    coarse_z = 2
                    fine_z = fine_xy
                else:
                    coarse_z = fine_xy
                    fine_z = 2
                # Now, the wavelet coefficients are in fine_z and the current coarse scale is in coarse_z

                # Do stuff
                self.handle_coefficients(fine_z, xy_scale, z_scale)

                z_scale_factor *= 2

            # Handle the smooth scale of the data
            # This corresponds to the total power if xyscale == self.xy_scales
            # and to the smooth part of the spectral decomposition otherwise
            self.handle_coefficients(coarse_z, xy_scale, self.z_scales)

            xy_scale_factor *= 2


    cdef void init_work(self):
        """
        Work array initializer. To be overriden if iterative reconstruction is desired.
        """

        cdef:
            float[:,:,:,:] work = self._work
            float[:,:,:] data = self._data

        work[0] = data


    cpdef handle_coefficients(self, int work_array, int xy_scale, int z_scale):
        """
        To be overridden by subclasses. Standard implementation only stores the coefficients
        in the current working directory.
        """
        np.save('coefficients_{0}_{1}.npy'.format(xy_scale, z_scale), self.work[work_array])


cdef class Denoise2D1DHard(WaveletDecomposition2D1D):
    """
    Subclass of WaveletDecomposition2D1D which implements a hard thresholding scheme for denoising.
    """

    cdef:
        np.ndarray _thresholds, _reconstruction
        bool _total_power, _xy_approx, _z_approx, _positivity

    def __init__(self, data, total_power=False, xy_approx=False, z_approx=False, xy_scales=-1, z_scales=-1):
        """
        Parameters
        ----------
        data : 3D ndarray
            The data to be denoised.

        total_power : boolean, optional
            Should the total power (i.e. the smooth scale) be part of the reconstruction?

        xy_approx, z_approx : boolean, optional
            Should the detail-approximation-coefficients (see Starck et al. 2009) be part of the reconstruction?
        """

        self.xy_approx = xy_approx
        self.total_power = total_power
        self.z_approx = z_approx

        self._reconstruction = np.zeros(data.shape, dtype=np.single)

        super(Denoise2D1DHard, self).__init__(data=data, xy_scales=xy_scales, z_scales=z_scales)


    property reconstruction:
        """
        ndarray containing the reconstruction of the data
        """
        def __get__(self):
            return self._reconstruction
        def __set__(self, value):
            value = np.array(value, dtype=np.single)
            if value.shape == self.data.shape:
                self._reconstruction = value
            else:
                raise ValueError("Reconstruction must have same shape as data")


    property thresholds:
        """
        ndarray containing the thresholds for each wavelet sub-band
        """
        def __get__(self):
            return self._thresholds
        def __set__(self, value):
            value = value.astype(np.single)
            if value.shape == (self.xy_scales + 1, self.z_scales + 1):
                self._thresholds = value.astype(np.single)
            else:
                raise ValueError(
                    "Thresholds must have shape %s"
                    % str(tuple(self.xy_scales + 1, self.z_scales + 1)))


    property total_power:
        """
        Wheather add total power to reconstruction
        """
        def __get__(self):
            return self._total_power
        def __set__(self, value):
            value = bool(value)
            self._total_power = value


    property xy_approx:
        """
        Wheather to use wavelet sub-bands belonging to the spatial
        approximation
        """
        def __get__(self):
            return self._xy_approx
        def __set__(self, value):
            value = bool(value)
            self._xy_approx = value


    property z_approx:
        """
        Wheather to use wavelet sub-bands belonging to the spectral
        approximation
        """
        def __get__(self):
            return self._z_approx
        def __set__(self, value):
            value = bool(value)
            self._z_approx = value


    property positivity:
        """
        Wheather to enforce a positivity constraint on the reconstruction
        """
        def __get__(self):
            return self._positivity
        def __set__(self, value):
            value = bool(value)
            self._positivity = value


    def decompose(self):

        cdef:
            float[:,:,:] reconstruction = self._reconstruction
            int i,j,k

        super(Denoise2D1DHard, self).decompose()

        if self._positivity:
            for i in range(reconstruction.shape[0]):
                for j in range(reconstruction.shape[1]):
                    for k in range(reconstruction.shape[2]):
                        if reconstruction[i,j,k] < 0:
                            reconstruction[i,j,k] = 0.


    cdef void init_work(self):

        cdef:
            int i,j,k
            float[:,:,:] data = self._data
            float[:,:,:] reconstruction = self._reconstruction
            float[:,:,:,:] work = self._work

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    work[0,i,j,k] = data[i,j,k] - reconstruction[i,j,k]


    cpdef handle_coefficients(self, int work_array, int xy_scale, int z_scale):

        if (xy_scale < self._xy_scales or self._xy_approx) and (z_scale < self._z_scales or self._z_approx):
            self.threshold_uniform(work_array, xy_scale, z_scale)
        elif z_scale == self._z_scales and xy_scale == self._xy_scales and self._total_power:
            self.add_total_power(work_array)


    cdef void threshold_uniform(self, int work_array, int xy_scale, int z_scale):
        cdef:
            float[:,:,:] data = self._data
            float[:,:,:] reconstruction = self._reconstruction
            float[:,:,:,:] work = self._work
            float[:,:] thresholds = self._thresholds
            int i, j, k

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if thresholds[xy_scale,z_scale] >= 0 and abs(work[work_array,i,j,k]) > thresholds[xy_scale,z_scale]:
                        reconstruction[i,j,k] += work[work_array,i,j,k]


    cdef void add_total_power(self, int work_array):
        cdef:
            float[:,:,:] reconstruction = self._reconstruction
            float[:,:,:,:] work = self._work
            int i,j,k

        for i in range(work.shape[0]):
            for j in range(work.shape[1]):
                for k in range(work.shape[2]):
                    reconstruction[i, j, k] += work[work_array, i, j, k]


cdef class Denoise2D1DHardMRS(Denoise2D1DHard):
    """
    Subclass of Denoise2D1DHard which adds the use of a multi-resolution support
    to the denoising process.
    """

    cdef:
        np.ndarray _mrs
        bool _fix_mrs

    def __init__(self, *args, **kwargs):

        super(Denoise2D1DHardMRS, self).__init__(*args, **kwargs)

        xy_scale_range = self.xy_scales
        if self.xy_approx:
            xy_scale_range += 1

        z_scale_range = self.z_scales
        if self.z_approx:
            z_scale_range += 1

        self._mrs = np.zeros((xy_scale_range, z_scale_range,) + self.data.shape, dtype=np.short)
        self._fix_mrs = False


    property mrs:
        """
        ndarray containing the multi-resolution support
        """
        def __get__(self):
            return self._mrs
        def __set__(self, value):
            value = np.array(value, dtype=np.short)
            if value.shape == self.mrs.shape:
                self._mrs = value


    property fix_mrs:
        """
        Wheather to keep the MRS fixed during iterative reconstruction.
        """
        def __get__(self):
            return self._fix_mrs
        def __set__(self, value):
            value = bool(value)
            self._fix_mrs = value


    cdef void threshold_uniform(self, int work_array, int xy_scale, int z_scale):
        cdef:
            float[:,:,:] data = self._data
            float[:,:,:] reconstruction = self._reconstruction
            float[:,:,:,:] work = self._work
            float[:,:] thresholds = self._thresholds
            short[:,:,:,:,:] mrs = self._mrs
            int i, j, k

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if self._fix_mrs:
                        if mrs[xy_scale, z_scale, i, j, k]:
                            reconstruction[i,j,k] += work[work_array,i,j,k]
                    elif mrs[xy_scale, z_scale, i, j, k] or (thresholds[xy_scale,z_scale] >= 0 and abs(work[work_array,i,j,k]) > thresholds[xy_scale,z_scale]):
                        reconstruction[i,j,k] += work[work_array,i,j,k]
                        mrs[xy_scale, z_scale, i, j, k] = 1


cdef class Denoise2D1DSoft(Denoise2D1DHard):
    """
    Subclass of Denoise2D1DHard which overrides the thresholding scheme to use
    the soft thresholding rule.
    """

    cdef void threshold_uniform(self, int work_array, int xy_scale, int z_scale):
        cdef:
            float[:,:,:] data = self._data
            float[:,:,:] reconstruction = self._reconstruction
            float[:,:,:,:] work = self._work
            float[:,:] thresholds = self._thresholds
            int i, j, k

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if thresholds[xy_scale,z_scale] >= 0 and abs(work[work_array,i,j,k]) > thresholds[xy_scale,z_scale]:
                        reconstruction[i,j,k] += abs(work[work_array,i,j,k] - thresholds[xy_scale,z_scale]) * sign(work[work_array,i,j,k])


cdef class WaveletDecomposition1D:
    cdef:
        np.ndarray _work, _data
        int _scales
        np.ndarray _mother_function

    def __init__(self, data, scales = -1):

        self._data = data.astype(np.single)
        self._work = np.empty((2,) + data.shape, dtype=np.single)

        if scales < 0:
            self._scales = np.ceil(np.log2(data.shape[0]))
        else:
            self._scales = scales

    property work:
        def __get__(self):
            return self._work

    property scales:
        def __get__(self):
            return self._scales

    property data:
        def __get__(self):
            return self._data

    property mother_function:
        def __get__(self):
            return self._mother_function
        def __set__(self, value):
            self._mother_function = np.array(value).astype(np.single)

    cdef void init_work(self):

        self._work[0] = self._data

    def decompose(self):

        cdef:
            float[:,:] work = self._work
            float[:] data = self._data
            int i, read_from, write_to, scale_factor, scale

        self.init_work()
        scale_factor = 1

        read_from = 0
        write_to = 1

        for scale in range(self._scales):

            ### Write smoothed version to write_to
            convolve_with_stepsize(work[read_from], work[write_to], self._mother_function, scale_factor)

            ### Subtract smoothed version from original data and store the coefficients in the original data array
            for i in range(data.size):
                work[read_from][i] = work[read_from][i] - work[write_to][i]

            ### Switch indices. The smoothed data is now in read_from for the next iteration
            read_from = (read_from + 1) % 2
            write_to = (write_to + 1) % 2

            self.handle_coefficients(write_to, scale)

            scale_factor *= 2

        self.handle_coefficients(read_from, self._scales)

    cpdef handle_coefficients(self, int work_array, int scale):
        np.save('coefficients_{0}.npy'.format(scale), self._work[work_array])


cdef class Denoise1DHardMRS(WaveletDecomposition1D):
    cdef:
        np.ndarray _mrs
        np.ndarray _reconstruction
        np.ndarray _thresholds
        bool _fix_mrs, _total_power

    def __init__(self, data, thresholds, scales=-1, total_power=False):

        super(Denoise1DHardMRS, self).__init__(data, scales)

        self._fix_mrs = False
        self._total_power = total_power

        self._mrs = np.zeros((self.scales,) + data.shape, dtype=np.short)
        self._reconstruction = np.zeros(data.shape, dtype=np.single)
        self._thresholds = np.array(thresholds).astype(np.single)

    property reconstruction:
        def __get__(self):
            return self._reconstruction
        def __set__(self, value):
            self._reconstruction = np.array(value).astype(np.single)

    property thresholds:
        def __get__(self):
            return self._thresholds
        def __set__(self, value):
            self._thresholds = value.astype(np.single)

    property total_power:
        def __get__(self):
            return self._total_power
        def __set__(self, value):
            self._total_power = value

    cdef void init_work(self):

        cdef:
            int i
            float[:,:] work = self._work
            float[:] data = self._data
            float[:] reconstruction = self._reconstruction

        for i in range(data.size):
            work[0,i] = data[i] - reconstruction[i]

    cpdef handle_coefficients(self, int work_array, int scale):

        cdef:
            int i
            float[:,:] work = self._work
            float[:] reconstruction = self._reconstruction
            float[:] thresholds = self._thresholds
            short[:,:] mrs = self._mrs

        if (scale == self._scales and self._total_power):
            for i in range(reconstruction.size):
                reconstruction[i] += work[work_array][i]

        elif (scale < self._scales):
            for i in range(reconstruction.size):
                if self._fix_mrs:
                    if mrs[scale,i]:
                        reconstruction[i] += work[work_array][i]

                elif mrs[scale,i] or (abs(work[work_array][i]) > thresholds[scale]):
                    reconstruction[i] += work[work_array][i]
                    mrs[scale,i] = 1


cdef class WaveletDecomposition3D:
    cdef:
        np.ndarray _work, _data
        int _scales
        np.ndarray _mother_function


    def __init__(self, data, scales = -1):

        self._data = data.astype(np.single)
        self._work = np.empty((3,) + data.shape, dtype=np.single)

        if scales < 0:
            self._scales = np.ceil(np.log2(np.amin(data.shape)))
        else:
            self._scales = scales

    property work:
        def __get__(self):
            return self._work

    property scales:
        def __get__(self):
            return self._scales

    property data:
        def __get__(self):
            return self._data

    property mother_function:
        def __get__(self):
            return self._mother_function
        def __set__(self, value):
            self._mother_function = np.array(value).astype(np.single)

    cdef void init_work(self):

        self._work[0] = self._data


    def decompose(self):

        cdef:
            float[:,:,:,:] work = self._work
            float[:,:,:] data = self._data
            int i, read_from, write_to, scale_factor, scale

        self.init_work()
        scale_factor = 1

        read_from = 0
        write_to = 1

        for scale in range(self._scales):

            ### Write smoothed version to write_to and avoid overwriting read_from
            convolve_x_with_stepsize(work[read_from], work[write_to], self._mother_function, scale_factor)
            convolve_y_with_stepsize(work[write_to], work[2], self._mother_function, scale_factor)
            convolve_z_with_stepsize(work[2], work[write_to], self._mother_function, scale_factor)

            ### Subtract smoothed version from original data and store the coefficients in the original data array
            inplace_diff(work[read_from], work[write_to])

            ### Switch indices. The smoothed data is now in read_from for the next iteration
            read_from = (read_from + 1) % 2
            write_to = (write_to + 1) % 2

            self.handle_coefficients(write_to, scale)

            scale_factor *= 2

        self.handle_coefficients(read_from, self._scales)

    cpdef handle_coefficients(self, int work_array, int scale):
        np.save('coefficients_{0}.npy'.format(scale), self._work[work_array])
