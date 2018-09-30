from __future__ import division
import warnings

import pylab as pl
from pylab import pi,sin,cos,exp


## Computational model parameters for FE problem
#
# This model generates element table and similar relevant
# parameters for the FE solver and optimization algorithm
class model():
	## Constructor to be called after containerBase constructor
	#
	#@param lx		Size of domain in x direction
	#@param lz		Size of domain in z direction
	#@param nely	(int) Number of elements in y direction
	#					 (nelx is calculated based on ly and nely)
	#@param Nm		(int) Number of positive terms in Fourier 
	#					 mode expansion 
	#@return			Model object
	def __init__(self,lx,lz,nelx,Nm=10):
		self.lx = lx
		self.lz = lz
		self.nelx = nelx
		self.Nm = int(Nm)
		self.NM = 2*self.Nm+1									#Total number of Fourier terms
		self.elmsize = self.lx/self.nelx						#Element size
		self.nelz = int(round(self.lz/self.lx*self.nelx))	#Number of elements in z
		tmp = self.nelz*self.elmsize
		if tmp != self.lz:
			warnings.warn("Model: Size of lz changed from {:.2f} to {:.2f} to comply with mesh".format(self.lz,tmp),Warning)
			self.lz = tmp
		self.ndof = (self.nelx+1)*(self.nelz+1)			#Degrees of freedom
		self.etab = self._getEtab()							#Elements table


	## Get parameters connected with the model object
	#
	#@return		A dictionary containing enough data to reconstruct the object from scratch
	def getParameters(self):
		params = {}
		params["lx"] = self.lx
		params["lz"] = self.lz
		params["nelx"] = self.nelx
		params["nelz"] = self.nelz
		params["Nm"] = self.Nm
		params["elmsize"] = self.elmsize
		return params

	def generateEmptyDesign(self):
		#Initialize dofs for elements in simulation domain
		x = pl.zeros((self.nelx,self.nelz))
		return x

	## Get element coordinates of domain
	#
	#@return		X,Z coordinates of all elements
	def getElementCoordinates(self):
		z = pl.arange(self.nelz)*self.elmsize+0.5*self.elmsize
		x = pl.arange(self.nelx) *self.elmsize+0.5*self.elmsize
		Z,X = pl.meshgrid(z[::-1],x[::-1])
		return X,Z

	## Get node coordinates of domain
	#
	#@return		X,Z coordinates of all elements
	def getNodeCoordinates(self):
		z = pl.arange(self.nelz+1)*self.elmsize
		x = pl.arange(self.nelx+1) *self.elmsize
		Z,X = pl.meshgrid(z[::-1],x[::-1])
		return X,Z

	#Get elements table
	#
	#@return A table containing 
	#				[x-index of element, 
	#				 y-index of element, 
	#				 4 indices to the  nodes making up the element]
	#				First index is 0 as according to std Python indexing
	#				May be flatten like this etab.reshape(M.nelx*M.nelz,4)
	def _getEtab(self):
		etab = pl.zeros([self.nelx,self.nelz,4],dtype=pl.int32)
		for elz in range(self.nelz):
			for elx in range(self.nelx):
				etab[elx,elz,0] = (self.nelx+1) * (elz  ) + elx
				etab[elx,elz,1] = (self.nelx+1) * (elz+1) + elx
				etab[elx,elz,2] = (self.nelx+1) * (elz+1) + elx+1
				etab[elx,elz,3] = (self.nelx+1) * (elz  ) + elx+1
		return etab
