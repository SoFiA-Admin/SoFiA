#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# distutils: language = c++
# distutils: sources = ModuleParametrisation.cpp Unit.cpp Measurement.cpp Source.cpp SourceCatalog.cpp helperFunctions.cpp MaskOptimization.cpp BusyFit.cpp DataCube.cpp MetaData.cpp Parametrization.cpp WorldCoordinateSystem.cpp

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from cparametrizer cimport *
from cython.operator cimport dereference as deref, preincrement as inc
from cpython cimport bool as python_bool
from libcpp.string cimport string
from cpython.version cimport PY_MAJOR_VERSION

unit_std = UNIT_STD
unit_exp = UNIT_EXP
unit_none = UNIT_NONE
unit_jy = UNIT_JY
unit_mjy = UNIT_MJY
unit_deg = UNIT_DEG
unit_arcmin = UNIT_ARCMIN
unit_arcsec = UNIT_ARCSEC
unit_mas = UNIT_MAS
unit_g = UNIT_G
unit_pc = UNIT_PC
unit_kpc = UNIT_KPC
unit_mpc = UNIT_MPC
unit_au = UNIT_AU
unit_ly = UNIT_LY
unit_min = UNIT_MIN
unit_h = UNIT_H
unit_a = UNIT_A
unit_erg = UNIT_ERG
unit_dyn = UNIT_DYN
unit_c = UNIT_C
unit_e = UNIT_E

unit_list = [
    unit_std, unit_exp, unit_none, unit_jy, unit_mjy, unit_g,
    unit_deg, unit_arcmin, unit_arcsec, unit_mas,
    unit_pc, unit_kpc, unit_mpc, unit_au, unit_ly,
    unit_min, unit_h, unit_a, unit_erg, unit_dyn, unit_c, unit_e
    ]

unit_names = [
    'unit_std', 'unit_exp', 'unit_none', 'unit_jy', 'unit_mjy', 'unit_g',
    'unit_deg', 'unit_arcmin', 'unit_arcsec', 'unit_mas',
    'unit_pc', 'unit_kpc', 'unit_mpc', 'unit_au', 'unit_ly',
    'unit_min', 'unit_h', 'unit_a', 'unit_erg', 'unit_dyn', 'unit_c', 'unit_e'
    ]


measurement_default = MEASUREMENT_DEFAULT
measurement_compact = MEASUREMENT_COMPACT
measurement_full = MEASUREMENT_FULL
measurement_unit = MEASUREMENT_UNIT

measurement_list = [
    measurement_default, measurement_compact, measurement_full, measurement_unit
    ]

measurement_names = [
    'measurement_default', 'measurement_compact', 'measurement_full', 'measurement_unit'
    ]

cdef class PyUnit(object):
    """Wrapper around (C++) Unit class"""

    def __cinit__(self, u=None):
        """Constructor PyUnit(u=None)

        u can be None (empty Unit is returned), a string or another PyUnit"""
        self.thisptr = new Unit()
        self.setU(u)

    def __dealloc__(self):
        del self.thisptr

    def setU(self, u=None):

        res = 0
        if u is None:
            self.clear()
        elif isinstance(u, basestring):
            res = self.thisptr.set(<string> u.encode('UTF-8'))
        elif isinstance(u, PyUnit):
            del self.thisptr
            self.thisptr = new Unit(deref((<PyUnit> u).thisptr))
        else:
            raise TypeError('PyUnit: Incompatible input type')

        if res == 1:
            raise ValueError('Units error')

    def asString(self, const unsigned int mode=unit_std):
        """mode can either be unit_std or unit_exp

        the latter does some pretty printing (utf8 needed)"""
        b = self.thisptr.printString(<const unsigned int> mode)
        return (b if PY_MAJOR_VERSION < 3 else b.decode('UTF-8'))

    def __repr__(self):
        return self.asString()

    def clear(self):
        self.thisptr.clear()

    unit = property(asString, setU, None, 'unit property')

    def getPrefix(self):
        return self.thisptr.getPrefix()

    def invert(self):
        self.thisptr.invert()

    def isEmpty(self):
        return self.thisptr.isEmpty()

    def isDefined(self):
        return self.thisptr.isDefined()

    def copy(self):
        return PyUnit(self)

    def __mul__(self, PyUnit other not None):
        assert isinstance(other, PyUnit), 'operation only allowed for PyUnit type'

        result = PyUnit(self)
        result.thisptr.mult_equal(deref((<PyUnit> other).thisptr))
        return result

    def __richcmp__(PyUnit self, PyUnit other not None, int op):
        assert isinstance(other, PyUnit), 'operation only allowed for PyUnit type'

        if op == 2:
            return self.thisptr.isequal(deref((<PyUnit> other).thisptr))
        if op == 3:
            return not self.thisptr.isequal(deref((<PyUnit> other).thisptr))
        else:
            raise NotImplementedError('PyUnit: requested operator not implemented')


cdef class PyMeasurement:
    """contains a var-name with value, uncertainty and a unit"""

    def __cinit__(self, name='noname', double value=0.0, double uncertainty=0.0, unit=''):
        """name can be str-type or another PyMeasurement
        (in the latter case, value/uncertainty/unit won't be handled!)"""
        self.thisptr = new Measurement[double]()
        self.setM(name, value, uncertainty, unit)

    def __dealloc__(self):
        del self.thisptr

    def setM(self, name='noname', double value=0.0, double uncertainty=0.0, unit=''):
        """Sets content of this Measurement, name can be str-type or another PyMeasurement
        (in the latter case, value/uncertainty/unit won't be handled!)

        If unit is an int/enum, the values are also multiplied by a conversion factor

        possible ints are:
        unit_std, unit_exp, unit_none, unit_jy, unit_mjy, unit_g,
        unit_deg, unit_arcmin, unit_arcsec, unit_mas,
        unit_pc, unit_kpc, unit_mpc, unit_au, unit_ly,
        unit_min, unit_h, unit_a, unit_erg, unit_dyn, unit_c, unit_e"""
        # NOTE, it would be great to use proper enums here, there is a backport
        # of the new python3.4 enum, called 'enum34'
        # should we add this to the requirements?
        if isinstance(name, PyMeasurement):
            del self.thisptr
            self.thisptr = new Measurement[double](deref((<PyMeasurement> name).thisptr))
        elif isinstance(name, basestring):
            self.setName(name)
            self.setValue(value)
            self.setUncertainty(uncertainty)
            self.setUnit(unit)
        else:
            raise TypeError('PyMeasurement: Incompatible input type (name must be str or PyMeasurement)')

    def setName(self, n):
        if not isinstance(n, basestring):
            raise TypeError('PyMeasurement: Incompatible input type (name must be str())')
        self.thisptr.setName(<string> n.encode('UTF-8'))

    def setValue(self, double v):
        self.thisptr.setValue(v)

    def setUncertainty(self, double u):
        self.thisptr.setUncertainty(u)

    def setUnit(self, u):
        if isinstance(u, int):
            if not u in unit_list:
                raise TypeError(
                    '\n'.join(
                        ['newUnit must be one of'] +
                        unit_names +
                        ['(if an integer/enum is used)']
                        )
                    )
            res = self.thisptr.setInt(
                <string> self.getName().encode('UTF-8'),
                <double> self.getValue(),
                <double> self.getUncertainty(),
                <unsigned int> u
                )
            if res == 1:
                raise ValueError('Units error')
        else:
            u = PyUnit(u)
            self.thisptr.setUnitUnit(deref((<PyUnit> u).thisptr))

    def getName(self):
        b = self.thisptr.getName()
        return (b if PY_MAJOR_VERSION < 3 else b.decode('UTF-8'))

    def getValue(self):
        return self.thisptr.getValue()

    def getUncertainty(self):
        return self.thisptr.getUncertainty()

    def getUnit(self):
        pu = PyUnit()
        pu.thisptr = new Unit(self.thisptr.getUnit())
        return pu

    name = property(getName, setName, None, 'name name')
    value = property(getValue, setValue, None, 'name value')
    uncertainty = property(getUncertainty, setUncertainty, None, 'name uncertainty')
    unit = property(getUnit, setUnit, None, 'name unit')

    def clear(self):
        self.thisptr.clear()

    def asString(
            self,
            unsigned int mode=measurement_default,
            int decimals=-1,
            python_bool scientific=False
            ):
        if not mode in measurement_list:
            raise TypeError(
                '\n'.join(
                    ['mode must be one of'] +
                    measurement_names
                    )
                )

        b = self.thisptr.printString(<unsigned int> mode, decimals, <bool> scientific)
        return (b if PY_MAJOR_VERSION < 3 else b.decode('UTF-8'))

    def __repr__(self):
        return self.asString()

    def convert(self, int unitenum):
        if not unitenum in unit_list:
            raise TypeError(
                '\n'.join(
                    ['newUnit must be one of'] +
                    unit_names +
                    ['(if an integer/enum is used)']
                    )
                )
        cdef double newValue = 0.
        cdef double newUncertainty = 0.
        res = self.thisptr.convert(newValue, newUncertainty, unitenum)
        if res == 1:
            raise ValueError('Units Incompatible')
        return newValue, newUncertainty

    def invert(self):
        res = self.thisptr.invert()
        if res == 1:
            raise ZeroDivisionError('float division by zero')

    def __richcmp__(PyMeasurement self, PyMeasurement other not None, int op):
        assert isinstance(other, PyMeasurement), 'operation only allowed for PyMeasurement type'

        # for some reason, the C++ implementation doesn't check for units being equal
        _unit1 = self.getUnit()
        _unit2 = other.getUnit()
        if _unit1 != _unit2:
            raise TypeError(
                'PyMeasurement: Incompatible units <' +
                _unit1.asString() + '> <' + _unit2.asString() + '>'
                )

        if op == 0:
            return self.thisptr.issmaller(deref((<PyMeasurement> other).thisptr))
        elif op == 1:
            return self.thisptr.issmallereq(deref((<PyMeasurement> other).thisptr))
        elif op == 2:
            return self.thisptr.isequal(deref((<PyMeasurement> other).thisptr))
        elif op == 3:
            return not self.thisptr.isequal(deref((<PyMeasurement> other).thisptr))
        elif op == 4:
            return self.thisptr.islarger(deref((<PyMeasurement> other).thisptr))
        elif op == 5:
            return self.thisptr.islargereq(deref((<PyMeasurement> other).thisptr))
        else:
            raise NotImplementedError('PyMeasurement: requested operator not implemented')

    def copy(self):
        return PyMeasurement(self)

    def __mul__(self, PyMeasurement other not None):
        assert isinstance(other, PyMeasurement), 'operation only allowed for PyMeasurement type'

        result = PyMeasurement(self)
        result.thisptr.mult_equal(deref((<PyMeasurement> other).thisptr))
        return result

    def __rdiv__(self, PyMeasurement other not None):
        assert isinstance(other, PyMeasurement), 'operation only allowed for PyMeasurement type'

        result = PyMeasurement(self)
        result.thisptr.div_equal(deref((<PyMeasurement> other).thisptr))
        return result

    def __add__(self, PyMeasurement other not None):
        assert isinstance(other, PyMeasurement), 'operation only allowed for PyMeasurement type'

        # this is also handled the C++ implementation but we want an exception raised
        _unit1 = self.getUnit()
        _unit2 = other.getUnit()
        if _unit1 != _unit2:
            raise TypeError(
                'PyMeasurement: Incompatible units [' +
                _unit1.asString() + '] [' + _unit2.asString() + ']'
                )

        result = PyMeasurement(self)
        result.thisptr.plus_equal(deref((<PyMeasurement> other).thisptr))
        return result

    def __sub__(self, PyMeasurement other not None):
        assert isinstance(other, PyMeasurement), 'operation only allowed for PyMeasurement type'

        # this is also handled the C++ implementation but we want an exception raised
        _unit1 = self.getUnit()
        _unit2 = other.getUnit()
        if _unit1 != _unit2:
            raise TypeError(
                'PyMeasurement: Incompatible units [' +
                _unit1.asString() + '] [' + _unit2.asString() + ']'
                )

        result = PyMeasurement(self)
        result.thisptr.minus_equal(deref((<PyMeasurement> other).thisptr))
        return result

    def __neg__(self):
        result = PyMeasurement(self)
        result.setValue(-self.getValue())
        return result


cdef class PySource:

    def __cinit__(self, s=None):
        self.thisptr = new Source()
        self.setParameters(s)

    def __dealloc__(self):
        del self.thisptr

    def isDefined(self):
        return self.thisptr.isDefined()

    def parameterDefined(self, s):
        assert isinstance(s, basestring), 'parameter name must be str'

        return self.thisptr.parameterDefined(<string> s.encode('UTF-8'))

    def setParameter(self, name='noname', double value=0.0, double uncertainty=0.0, unit=''):
        """name can be str-type or another PyMeasurement
        (in the latter case, value/uncertainty/unit won't be handled!)"""
        m = PyMeasurement(name, value, uncertainty, unit)
        self.thisptr.setParameter(deref((<PyMeasurement> m).thisptr))

    def getParameter(self, s):
        assert isinstance(s, basestring), 'parameter name must be str'

        pm = PyMeasurement()
        pm.thisptr = new Measurement[double](self.thisptr.getParameterMeasurement(<string> s.encode('UTF-8')))
        if pm.getName() == 'notfound':
            raise KeyError('PySource: Parameter not found')
        return pm

    def setSourceID(self, unsigned long sid):
        """Note: floats will be silently converted (truncated) to Int"""
        self.thisptr.setSourceID(sid)

    def getSourceID(self):
        return self.thisptr.getSourceID()

    def setSourceName(self, s):
        assert isinstance(s, basestring), 'source name must be str'
        self.thisptr.setSourceName(<string> s.encode('UTF-8'))

    def getSourceName(self):
        b = self.thisptr.getSourceName()
        return (b if PY_MAJOR_VERSION < 3 else b.decode('UTF-8'))

    ID = property(getSourceID, setSourceID, None, 'ID property')
    name = property(getSourceName, setSourceName, None, 'Source name property')

    def getParameters(self):
        pdict = {}
        cdef map[string, Measurement[double]].iterator mapiter
        cdef map[string, Measurement[double]] pmap
        pmap = self.thisptr.getParameters()
        mapiter = pmap.begin()
        while mapiter != pmap.end():
            pm = PyMeasurement()
            pm.thisptr = new Measurement[double](deref(mapiter).second)
            b = deref(mapiter).first
            pdict[(b if PY_MAJOR_VERSION < 3 else b.decode('UTF-8'))] = pm
            inc(mapiter)
        return pdict

    def setParameters(self, parameters):
        """parameters can be None (source gets cleared), another PySource instance or a dictionary
        of PyMeasurements"""
        self.clear()
        if parameters is None:
            return
        elif isinstance(parameters, PySource):
            pDict = parameters.getParameters()
        elif isinstance(parameters, dict):
            assert all([isinstance(m, PyMeasurement) for m in parameters.values()]), \
                'any dictionary value is not a PyMeasurement'
            pDict = parameters
        else:
            raise TypeError('PySource: Incompatible input type')

        for key in pDict:
            self.setParameter(pDict[key])

    def updateParameters(self, parameters):
        """parameters can be None (source gets cleared), another PySource instance or a dictionary
        of PyMeasurements"""
        if parameters is None:
            return
        elif isinstance(parameters, PySource):
            pDictNew = parameters.getParameters()
        elif isinstance(parameters, dict):
            assert all([isinstance(m, PyMeasurement) for m in parameters.values()]), \
                'any dictionary value is not a PyMeasurement'
            pDictNew = parameters
        else:
            raise TypeError('PySource: Incompatible input type')

        pDict = self.getParameters()
        pDict.update(pDictNew)

        self.clear()
        for key in pDict:
            self.setParameter(pDict[key])

    def clear(self):
        self.thisptr.clear()

    def copy(self):
        _s = PySource(self)
        _s.ID = self.ID
        _s.name = self.name
        return _s


cdef class PySourceCatalog:

    def __cinit__(self, sc=None):
        self.thisptr = new SourceCatalog()
        self.setSources(sc)

    def __dealloc__(self):
        del self.thisptr

    def readDuchampFile(self, filename):
        self.thisptr.readDuchampFile(<string> filename.encode('UTF-8'))

    def insert(self, PySource s, python_bool doCheck=True):
        sID = s.getSourceID()
        if doCheck:
            if sID in self.getSourceIDs():
                raise ValueError('Source ID ({0:d}) already present, please use update()'.format(sID))
        res = self.thisptr.insert(deref((<PySource> s).thisptr))
        assert res == 0, 'This should never happen'

    def update(self, unsigned long sID, PySource s, python_bool doCheck=True):
        assert sID == s.getSourceID(), 'sID from function call doesn\'t match that of PySource'
        if doCheck:
            if sID not in self.getSourceIDs():
                raise ValueError('Source ID ({0:d}) not present, please use insert()'.format(sID))
        res = self.thisptr.update(sID, deref((<PySource> s).thisptr))
        assert res == 0, 'This should never happen'

    def getSources(self):
        pdict = {}
        cdef map[unsigned long, Source].iterator mapiter
        cdef map[unsigned long, Source] pmap
        pmap = self.thisptr.getSources()
        mapiter = pmap.begin()
        while mapiter != pmap.end():
            pm = PySource()
            pm.thisptr = new Source(deref(mapiter).second)
            pdict[deref(mapiter).first] = pm
            inc(mapiter)
        return pdict

    def getSourceIDs(self):
        plist = []
        cdef map[unsigned long, Source].iterator mapiter
        cdef map[unsigned long, Source] pmap
        pmap = self.thisptr.getSources()
        mapiter = pmap.begin()
        while mapiter != pmap.end():
            plist.append(deref(mapiter).first)
            inc(mapiter)
        return sorted(plist)

    def setSources(self, sources):
        """sources maybe be None (sourceCatalog gets cleared) a dictionary or another
        PySourceCatalog"""

        self.clear()
        if sources is None:
            return
        elif isinstance(sources, PySourceCatalog):
            sDict = sources.getSources()
        elif isinstance(sources, dict):
            assert all([isinstance(s, PySource) for s in sources.values()]), \
                'any dictionary value is not a PySource'
            sDict = sources
        else:
            raise TypeError('PySourceCatalog: Incompatible input type')

        # check whether any new dict-ID doesn't match the source-ID
        assert all([
            key == sDict[key].getSourceID() for key in sDict
            ]), 'All dictionary keys must equal their source\'s ID!'

        for key in sDict:
            self.insert(sDict[key], False)  # no check needed, since catalog was cleared

    def updateSources(self, sources, python_bool warn_on_duplicate=True):
        """insert sources (dictionary or another PySourceCatalog) into catalog
        (without clearing it first

        if warn_on_duplicate is True, an AssertionError will be thrown, if you try
        to insert any source, the ID of which is already present in the catalog"""

        if isinstance(sources, PySourceCatalog):
            sDict = sources.getSources()
        elif isinstance(sources, dict):
            assert all([isinstance(s, PySource) for s in sources.values()]), \
                'any dictionary value is not a PySource'
            sDict = sources
        else:
            raise TypeError('PySourceCatalog: Incompatible input type')

        # check whether any new dict-ID doesn't match the source-ID
        assert all([
            key == sDict[key].getSourceID() for key in sDict
            ]), 'All dictionary keys must equal their source\'s ID!'

        _presentIDs = self.getSourceIDs()
        _newIDs = list(sDict.keys())
        _needsCheck = False
        if len(set(_presentIDs).intersection(set(_newIDs))) > 0:
            _needsCheck = True
            if warn_on_duplicate:
                raise AssertionError('one or more of the IDs already present in catalog')


        for key in sDict:
            if _needsCheck:
                sID = sDict[key].getSourceID()
                if sID in _presentIDs:
                    print('warning: ID', sID, 'already present, will overwrite')
                    self.update(sID, sDict[key])
                else:
                    self.insert(sDict[key], False)  # no check needed, have already handled it
            else:
                self.insert(sDict[key], False)  # no check needed, have already handled it

    def clear(self):
        self.thisptr.clear()

    def copy(self):
        return PySourceCatalog(self)

cdef class PyModuleParametrisation:

    def __cinit__(self):
        self.thisptr = new ModuleParametrisation()

    def __dealloc__(self):
        del self.thisptr

    def run(
            self,
            np.ndarray[float, ndim=3] datacube,
            np.ndarray[short, ndim=3] maskcube,
            PySourceCatalog initCatalog,
            ):
        """Note: for each source id there must be a corresponding region (subcube) in the
        maskcube, having pixel values that equal source id"""
        assert datacube.dtype is np.dtype('float32'), 'data must have type float (np.float32 aka <f4)'
        assert maskcube.dtype is np.dtype('int16'), 'mask must have type short (np.int16 aka <i2)'
        assert datacube.ndim == 3

        cdef long dz = datacube.shape[0]
        cdef long dy = datacube.shape[1]
        cdef long dx = datacube.shape[2]

        initCatalogPtr = new SourceCatalog(deref((<PySourceCatalog> initCatalog).thisptr))
        self.thisptr.run(
            <float*> datacube.data,
            <short*> maskcube.data,
            dx,
            dy,
            dz,
            deref(initCatalogPtr)
            )

    def getCatalog(self):
        c = PySourceCatalog()
        c.thisptr = new SourceCatalog(self.thisptr.getCatalog())
        return c

    def setFlags(self, python_bool doMaskOptimization, python_bool doBusyFitting):
        self.thisptr.setFlags(<bool> doMaskOptimization, <bool> doBusyFitting)
