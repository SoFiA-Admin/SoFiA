# Cythonize .pyx if Cython is available
try:
	from Cython.Build import cythonize

	pyx_src = [
		'CNHI.pyx'
	]

	cythonize(pyx_src, compiler_directives={'language_level' : "3"})

except ImportError:
	pass
