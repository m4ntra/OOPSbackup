from __future__ import division
import warnings

import pylab as pl
from pylab import pi,sin,cos,exp,sqrt

#Ugly hack to import materials module... Needs to be fixed
import os,sys
PATH = os.path.split(os.path.realpath(__file__))[0]+"/../"
sys.path.append(PATH)
from materials.material import loadMaterial
#from ..materials.material import loadMaterial


## Physics class
#
#Class holding physics based parameters for the model class
class physics():
	## Constructor
	#
	#@param model		The model class to which the physics should be imposed
	#@param lam			Free space wavelength of simulation
	#@param nIn			Refractive index at top interface
	#@param nOut		Refractive index at bottom interface
	#@param thetaIn	Incident angle of incoming plane wave (rad.)
	#@param pol			Polarization of inc. wave, either 'Ey' or 'Hy'	
	#@param interpolationType	Use x values directly in interpolate method or replace by materials:
	#									'inputBased' or 'materialBased' (default) respectively
	#@param nInName	Assign a custom string name to top interface material
	#@param nOutName	Assign a custom string name to bottom interface material
	#@return				Model object
	def __init__(self,model,lam,nIn=1.,nOut=1.,thetaIn=0,pol='Ey',interpolationType='materialBased',
					nInName="top interface",nOutName="bottom interface"):
		M = model
		self.M = M
		self.lam = lam
		self.nIn  = nIn
		self.nOut = nOut
		self.thetaIn = thetaIn
		self.pol = pol
		self.interpolationType = interpolationType
		self.materials = [self.nIn,self.nOut]
		self.materialNames = [nInName,nOutName]

		if self.nIn.imag != 0:
			warnings.warn("Input refractive index HAS to be real! Expect wrong results.",Warning)
		if self.nOut.imag != 0:
			pass
			#warnings.warn("Warning, it is not thorougly tested if a complex epsOut works",Warning)
			#Seems to work.. but I guess transmission parameters make little meaning

		#####################
		##Derived parameters
		#####################
		self.k0    = 2*pi/self.lam			#Wave number			
		self.freq = self.k0/(2*pi)			#Frequency

		self.epsIn = self.nIn**2
		self.epsOut = self.nOut**2

		self.kIn   = self.nIn*self.k0		#Wavenumber for top material
		self.kOut  = self.nOut*self.k0	#Wavenumber for bottom material

		wavefrac = self.lam/(M.elmsize*max(self.nIn,self.nOut))
		if wavefrac < 10:
			warnings.warn("You are using less than 10 elm per wavelength (may be okay"+
								"for highly absorptive/reflective materials",Warning)
		if wavefrac < 6:
			warnings.warn("SERIOUSLY?? You are using less than 6 elm per wavelength (may be okay"+
								"for highly absorptive/reflective materials",Warning)

		self.kInx = self.kIn*sin(self.thetaIn)		#Inc wave vector component
		self.kInz = self.kIn*cos(self.thetaIn)		# -||-

		##################################
		#Precalculated sizes for FE solver
		##################################
		self.alpha0 = self.kInx
		self.alpha = pl.zeros(M.NM)
		#print "Det ser ud til at chi er defineret forkert i Diaz vs Dossou (den skal ikke konjugeres)"
		self.chiIn = pl.zeros(M.NM,dtype='complex')
		self.chiOut = pl.zeros(M.NM,dtype='complex')
		for m in range(-M.Nm,M.Nm+1):
			idx = m+M.Nm
			self.alpha[idx] = self.alpha0+2*pi*m/M.lx

			self.chiIn[idx] = pl.conj(sqrt(0.j+self.kIn**2-self.alpha[idx]**2))
			self.chiOut[idx] = pl.conj(sqrt(0.j+self.kOut**2-self.alpha[idx]**2))

		self.thetaModesIn  = pl.arcsin(0.j+self.alpha/self.kIn)
		self.thetaModesOut = pl.arcsin(0.j+self.alpha/self.kOut)
		self.propModesIn = abs(self.thetaModesIn.imag) < 1e-8	#Slice of incoming propagating modes
		self.propModesOut = abs(self.thetaModesOut.imag) < 1e-8	#Slice of outgoing propagating modes	


	## Get parameters connected with the physics object
	#
	#@return		A dictionary containing enough data to reconstruct the object from scratch
	#				- exluding the model parameters!
	def getParameters(self):
		params = {}
		params["lam"] = self.lam
		params["nIn"] = self.nIn
		params["nOut"] = self.nOut
		params["thetaIn"] = self.thetaIn
		params["pol"] = self.pol
		params["chiIn"] = self.chiIn
		params["chiOut"] = self.chiOut
		params["thetaModesIn"] = self.thetaModesIn
		params["thetaModesOut"] = self.thetaModesOut
		params["propModesIn"] = self.propModesIn
		params["propModesOut"] = self.propModesOut
		params["materials"] = self.materials
		params["materialNames"] = self.materialNames
		return params

	## Define a new material
	#
	#@value	Either a string referring to a text file in the materials module folder (without .txt)
	#			or a constant (complex) value specifying the refractive index (complex values are 
	#			negative for lossy materials)
	#@name	Text string to label the material with. If non is specified and the other parameter
	#			is a string referring to a file, then that string will be used.
	#
	#@return	The index the new material has been given. When setting this index in a design field,
	#			the will be replaced with the material value.
	def newMaterial(self,value,name=None):
		n = loadMaterial(value,self.lam)
		if name==None:
			if type(value) == str:
				name = value
			else:
				name = "n="+str(value)
		self.materials += [n]
		self.materialNames += [name]
		return len(self.materials)-1


	## Interpolate
	#
	#@param x		Design field to be converted to input for the FE solver. If interpolationType
	#					is set to 'materialBased', then the values are replaced by materials specified
	#					by the user. If 'inputBased', then values in x are interpreted directly as refractive
	#					indices.
	#
	#@return			A, B as they are expected to be used by the solver.
	def interpolate(self,x):
		if x.shape != (self.M.nelx,self.M.nelz):
			print(x.shape)
			print(self.M.nelx,self.M.nelz)
			raise Exception("The input design field does not match the shape expected by the model object")
		if not (self.pol in ['Ey','Hy']):
			raise ValueError('The polarisation has to be set to either Ey or Hy.')

		##Set material parameters depending on whether E or H field is solved for
		# Starting with the boundaries
		if self.pol == 'Ey': 
			self.AIn  = 1
			self.AOut = 1
			self.BIn  = _nToEps(self.nIn)
			self.BOut = _nToEps(self.nOut)
		elif self.pol == 'Hy':
			self.AIn  = 1/_nToEps(self.nIn)
			self.AOut = 1/_nToEps(self.nOut)
			self.BIn  = 1
			self.BOut = 1

		A = pl.ones(x.shape,dtype='complex')
		B = pl.ones(x.shape,dtype='complex')

		if self.interpolationType == 'materialBased':
			if self.pol == 'Ey': 
				for i in range(len(self.materials)):
					B[x==i] = _nToEps(self.materials[i])
			elif self.pol == 'Hy':
				for i in range(len(self.materials)):
					A[x==i] = 1./_nToEps(self.materials[i])

		elif self.interpolationType == 'inputBased':
			if self.pol == 'Ey': 
				B[:] = _nToEps(x)
			elif self.pol == 'Hy':
				A[:] = 1./_nToEps(x)
		else:
			raise ValueError('interpolationType has to be either materialBased or inputBased')

		if abs(A[:,0]-self.AIn).max() or abs(B[:,0]-self.BIn).max():
			warnings.warn("The material parameters at the top interface does not correspond "+
								"to the specified nIn. Results will probably be flawed",Warning)
		if abs(A[:,-1]-self.AOut).max() or abs(B[:,-1]-self.BOut).max():
			print(abs(A[:,-1]-self.AOut).max())
			print(abs(B[:,-1]-self.BOut).max())
			#print(B[:,-1])
			#print(self.BOut)
			warnings.warn("The material parameters at the bottom interface does not correspond "+
								"to the specified nOut. Results will probably be flawed",Warning)
		return A,B


def _nToEps(n):
	return (n.real**2-n.imag**2) - 2.j*n.real*n.imag
