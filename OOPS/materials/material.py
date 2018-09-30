from __future__ import division
import pylab as pl
import os

PATH = os.path.split(os.path.realpath(__file__))[0]+"/"

def loadMaterial(name,llam):
	## Loads a given material from a list of text files found in the materials folder
	#
	# @name		The name (stirng) of the text file without the .txt exetension
	#				!If a non-string is given as input (i.e. a constant), that value is just returned
	# @llam		The list of wavelengths (in nm) that the material parameters should be evaluated at
	#
	# @return	Complex refractive index (absorbing material having negative imaginary part)
	if not type(name) == str:
		return name
	f = open(PATH+name+".txt",'r')
	f.next()

	lam = []
	n = []
	k = []
	for line in f:
		try:
			tmp = [float(elm) for elm in line.split()]
			lam += [tmp[0]]
			n += [tmp[1]]
			k += [tmp[2]]
		except:
			break

	n = pl.interp(llam,lam,n)
	k = pl.interp(llam,lam,k)
	return n - 1.j*k
