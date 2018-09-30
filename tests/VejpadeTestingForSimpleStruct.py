from __future__ import division
import pylab as pl

from pylab import pi,sin,cos,exp,tanh,sqrt

from scipy.sparse import coo_matrix, csc_matrix

from scipy.sparse.linalg import splu

import os,sys
PATH = os.path.split(os.path.realpath(__file__))[0]+"/../"
sys.path.append(PATH)

import OOPS
from OOPS.DossouSolver2D.solver import FE,calcRT,calcScatteringMatrix
from OOPS.DossouSolver2D.physicsClass import physics
from OOPS.DossouSolver2D.pade import pade
from OOPS.DossouSolver2D.modelClass import model
from auxiliary.dataHandling import loadDicts,saveDicts,dictSlice


nelx = 40
def main():
	lam0 = 500
	dlam = 150
	llam = pl.linspace(lam0-dlam,lam0+dlam,41)
	#R_0 = normal calculation, R_0_p = pade calculation
	R0 = []
	R0p = []
		
	#Plot reference
	if 1:
		for lam in llam:
			R = calcNormal(lam)
			R0 += [R]
	pl.plot(llam,R0,'r',lw=2,label = 'standard')
	llampade,Rpade = calcpade(lam0,dlam)
	pl.plot(llampade,Rpade,'b',lw=2,label = 'Pade')
	pl.title('Simulated reponse at normal incidence')
	pl.xlabel('wavelength (nm)')
	pl.ylabel('relative intensity')
	pl.legend(loc="upper right")
	pl.ylim(0,1)
	pl.savefig('pade.png')


## Erros in calc:
#
#	The model is scaled with the wavelength and the wavelength remains constant!

#lam is the wavelength
def calcNormal(lam):	
	Mres,Pres,xres = prepareMPx(lam)
	res = FE(Mres,Pres,xres,printResults=False)
	mR,thetaR,R,mT,thetaT,T = calcRT(Mres,Pres,res["r"],res["t"])
	return R[mR==0]

def calcpade(lam0,dlam):
	Mres,Pres,xres = prepareMPx(lam0)
	return pade(2,lam0,dlam, Mres, Pres, xres)


def prepareMPx(lam):
	thetain = -0/180*pl.pi
	nIn = 1.0
	nMaterial = 2.
	nOut = 1.
	pol = 'Ey'

	Mres = model(lx=320,lz=320,nelx=nelx)
	Pres = physics(Mres,lam,nIn,nOut,thetaIn=thetain,pol=pol)
	materialRes = Pres.newMaterial(nMaterial)
	xres = Mres.generateEmptyDesign()
	#Try vary the radius between 20 and 140
	makeCircle(xres,Mres,160,100,materialRes)
	if 1:	#Show design
		pl.imshow(xres.T,interpolation='none')
		pl.savefig('padeuc.png')
		exit()
	return Mres,Pres,xres


def makeCircle(x,M,z0,r,mat):
	X,Z = M.getElementCoordinates()
	x0 = M.lx/2
	idx = pl.sqrt((X-x0)**2+(Z-z0)**2) <r
	x[idx] = mat

def makeSlab(x,M,zstart,zstop,mat):
	X,Z = M.getElementCoordinates()
	idx = ((Z-zstart)>=0) * ((Z-zstop)<=0)
	x[idx] = mat



if __name__ == "__main__":
	main()
