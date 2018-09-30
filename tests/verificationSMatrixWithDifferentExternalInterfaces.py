from __future__ import division
import pylab as pl

from pylab import pi,sin,cos,exp,tanh,sqrt

import os,sys
PATH = os.path.split(os.path.realpath(__file__))[0]+"/../"
sys.path.append(PATH)

import OOPS
from OOPS.DossouSolver2D.solver import FE,calcRT,calcScatteringMatrix
from OOPS.DossouSolver2D.physicsClass import physics
from OOPS.DossouSolver2D.modelClass import model
from auxiliary.dataHandling import loadDicts,saveDicts,dictSlice

def main():
	lam = 505
	thetain = -30/180*pl.pi
	nIn = 1.0
	nMaterial = 12.
	nOut = 2.
	pol = 'Ey'

	##Reference model (one circle + one burried circle)
	Mres = model(lx=lam,lz=2*lam,nelx=40)
	Pres = physics(Mres,lam,nIn,nOut,thetaIn=thetain,pol=pol)
	if 1:	#Put to zero to avoid re-simulation and use last saved results
		materialRes = Pres.newMaterial(nMaterial)
		materialOut = Pres.newMaterial(nOut)

		xres = Mres.generateEmptyDesign()
		makeSlab(xres,Mres,0,0.5*lam,materialOut)	#make bottom part of domain nOut
		makeCircle(xres,Mres,250,200,materialRes)
		makeCircle(xres,Mres,250+lam,200,materialRes)

		if 1:	#Show design
			pl.imshow(xres.T,interpolation='none')
			pl.colorbar()
			pl.show()
			exit()

		res = FE(Mres,Pres,xres)
		saveDicts("SmatrixRIReference.h5",res,'w')
	else:
		res = loadDicts("SmatrixRIReference.h5")[0]
	mR,thetaR,R,mT,thetaT,T = calcRT(Mres,Pres,res["r"],res["t"])
	Rres = R
	Tres = T

	##Scattering matrix model 1 (one circle)
	M = model(lx=lam,lz=lam,nelx=40)
		#Notice how nOut here is the same as nIn
	P = physics(M,lam,nIn,nIn,thetaIn=thetain,pol=pol)
	material = P.newMaterial(nMaterial)
	x = M.generateEmptyDesign()
	makeCircle(x,M,250,200,material)
	if 0:	#Show design
		pl.imshow(x.T,interpolation='none')
		pl.colorbar()
		pl.show()
		exit()
	res = calcScatteringMatrix(M,P,x)
	S1 = RTtoS(res["RIn"],res["TIn"],res["TOut"],res["ROut"])

	##Scattering matrix model 2 (one burried circle)
	M = model(lx=lam,lz=lam,nelx=40)
	P = physics(M,lam,nIn,nOut,thetaIn=thetain,pol=pol)
	material = P.newMaterial(nMaterial)
	materialOut = P.newMaterial(nOut)
	x = M.generateEmptyDesign()
	makeSlab(x,M,0,0.5*lam,materialOut)
	makeCircle(x,M,250,200,material)
	if 0:	#Show design
		pl.imshow(x.T,interpolation='none')
		pl.colorbar()
		pl.show()
		exit()
	res = calcScatteringMatrix(M,P,x)
	S2 = RTtoS(res["RIn"],res["TIn"],res["TOut"],res["ROut"])




	#Stack the two scattering elements
	Stot = stackElements(S1,S2)

	#Define incident wave as a plane wave
	einc = pl.zeros((2*M.NM,1),dtype='complex').view(pl.matrix)
	einc[M.NM//2,0] = 1.
	tmp = Stot*einc
	rnew = tmp[:M.NM].view(pl.ndarray).flatten()
	tnew = tmp[M.NM:].view(pl.ndarray).flatten()
	mR,thetaR,Rnew,mT,thetaT,Tnew = calcRT(Mres,Pres,rnew,tnew)
	print("-"*32)
	print("Error in reflection  : {:.2e}".format(abs(Rnew-Rres).max()))
	print("Error in transmission: {:.2e}".format(abs(Tnew-Tres).max()))
	#print("(note: since the significant numbers are in general between 0.01 and 1")
	#print(" we required a much lower error (1e-6 or better) to confirm that it works)")



def RTtoS(RIn,TIn,TOut,ROut):
	S = pl.bmat(   [[RIn,   TOut],
						[TIn,   ROut]])
	return S

def StoRT(S):
	NM = S.shape[0]//2
	RIn = S[:NM,:NM]
	TIn = S[NM:,:NM]
	ROut = S[NM:,NM:]
	TOut = S[:NM,NM:]
	return RIn,TIn,TOut,ROut


def stackElements(S1,S2):
	# Reursive relations, eq (59-62) in "A combined three-dimensional finite element and scattering matrix 
	#  method for the analysis of plane wave diffraction by bi-periodic, multilayered structures" 
	#  (Dossou 2012a)
	RIn1,TIn1,TOut1,ROut1 = StoRT(S1)
	RIn2,TIn2,TOut2,ROut2 = StoRT(S2)
	I = pl.identity(RIn1.shape[0])
	RIn = RIn1+TOut1*RIn2*pl.inv(I-ROut1*RIn2)*TIn1
	TIn = TIn2*pl.inv(I-ROut1*RIn2)*TIn1
	ROut = ROut2+TIn2*ROut1*pl.inv(I-RIn2*ROut1)*TOut2
	TOut = TOut1*pl.inv(I-RIn2*ROut1)*TOut2
	return RTtoS(RIn,TIn,TOut,ROut)


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
