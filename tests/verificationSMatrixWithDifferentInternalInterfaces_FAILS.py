from __future__ import division
import pylab as pl

from pylab import pi,sin,cos,exp,tanh,sqrt

import os,sys
PATH = os.path.split(os.path.realpath(__file__))[0]+"/../"
sys.path.append(PATH)

import OOPS
#from OOPS.DossouSolver2D.solver import FE,calcRT,calcScatteringMatrix
from OOPS.DossouSolver2D.solver import FE,calcScatteringMatrix
from OOPS.DossouSolver2D.physicsClass import physics
from OOPS.DossouSolver2D.modelClass import model
from auxiliary.dataHandling import loadDicts,saveDicts,dictSlice


def calcRT(M,P,r,t):
	m = pl.arange(-M.Nm,M.Nm+1)

	idx = P.propModesIn
	mR = m[idx]
	thetaR =  P.thetaModesIn[idx].real
	R = (P.chiIn[idx]/P.chiIn[M.Nm]).real*abs(r[idx])**2

	idx = P.propModesOut
	mT = m[idx]
	thetaT = P.thetaModesOut[idx].real
	T = (P.chiOut[idx]/P.chiIn[M.Nm]).real*abs(t[idx])**2
	if P.pol == 'Hy':
		T = T*(P.nIn/P.nOut)**2 

	return mR,thetaR,R,mT,thetaT,T

def main():
	lam = 505
	thetain = -30/180*pl.pi
	nIn = 2.8
	nMaterial1 = 5.9
	nMaterial2 = 4.2
	#nMaterial2 = nIn	#When nMaterial2 == nIn, the methods works
	nOut = 4.3 #If nOut<nIn, we can have values in the S matrix exceeding 1
	nOut = 2.1
	#nOut = nMaterial2
	pol = 'Ey'
	nelx = 80

	##Reference model (one circle + one burried circle)
	Mres = model(lx=lam,lz=2*lam,nelx=nelx)
	Pres = physics(Mres,lam,nIn,nOut,thetaIn=thetain,pol=pol)
	if 1:	#Put to zero to avoid re-simulation and use last saved results
		materialRes1 = Pres.newMaterial(nMaterial1)
		materialRes2 = Pres.newMaterial(nMaterial2)
		materialResOut = Pres.newMaterial(nOut)

		xres = Mres.generateEmptyDesign()
		makeSlab(xres,Mres,0,0.5*lam,materialResOut)	
		makeSlab(xres,Mres,0.5*lam,1.5*lam,materialRes2)
		makeCircle(xres,Mres,250,200,materialRes1)
		makeCircle(xres,Mres,250+lam,200,materialRes1)

		if 0:	#Show design
			pl.imshow(xres.T,interpolation='none',vmin=0,vmax=4)
			pl.colorbar()
			pl.savefig("full_structure.png",bbox_inches='tight')
			pl.show()
			exit()

		res = FE(Mres,Pres,xres)
		saveDicts("SmatrixRIInternalReference.h5",res,'w')
	else:
		res = loadDicts("SmatrixRIInternalReference.h5")[0]
	mR,thetaR,R,mT,thetaT,T = calcRT(Mres,Pres,res["r"],res["t"])
	Rres = R
	Tres = T

	##Scattering matrix model 1 (one burried circle)
	M = model(lx=lam,lz=lam,nelx=nelx)
		#Notice how nOut here is nMaterial2
	P = physics(M,lam,nIn,nMaterial2,thetaIn=thetain,pol=pol)
	material1 = P.newMaterial(nMaterial1)
	material2 = P.newMaterial(nMaterial2)
	x = M.generateEmptyDesign()
	makeSlab(x,M,0,.5*lam,material2)
	makeCircle(x,M,250,200,material1)
	if 0:	#Show design
		pl.imshow(x.T,interpolation='none',vmin=0,vmax=4)
		pl.colorbar()
		pl.savefig("upper_half.png",bbox_inches='tight')
		pl.show()
		exit()
	res = calcScatteringMatrix(M,P,x)


	S1 = RTtoS(res["RIn"],res["TIn"],res["TOut"],res["ROut"])
	matL = pl.bmat([[ pl.diag(pl.sqrt(P.chiIn)), pl.zeros((M.NM,M.NM))],
						 [ pl.zeros((M.NM,M.NM)), pl.diag(pl.sqrt(P.chiOut)) ]])
	matR = pl.bmat([[ pl.diag(1/pl.sqrt(P.chiIn)), pl.zeros((M.NM,M.NM))],
						 [ pl.zeros((M.NM,M.NM)), pl.diag(1/pl.sqrt(P.chiOut)) ]])
	S1real = matL*S1*matR



	##Scattering matrix model 2 (one burried circle)
	M = model(lx=lam,lz=lam,nelx=nelx)
	thetainNew = pl.arcsin(P.nIn/P.nOut*pl.sin(thetain))
	P = physics(M,lam,nMaterial2,nOut,thetaIn=thetainNew,pol=pol)
	material1 = P.newMaterial(nMaterial1)
	material2 = P.newMaterial(nMaterial2)
	materialOut = P.newMaterial(nOut)
	x = M.generateEmptyDesign()
	makeSlab(x,M,.5*lam,lam,material2)
	makeSlab(x,M,0,.5*lam,materialOut)
	makeCircle(x,M,250,200,material1)
	if 0:	#Show design
		pl.imshow(x.T,interpolation='none',vmin=0,vmax=4)
		pl.colorbar()
		pl.savefig("lower_half.png",bbox_inches='tight')
		pl.show()
		exit()
	res = calcScatteringMatrix(M,P,x)
	S2 = RTtoS(res["RIn"],res["TIn"],res["TOut"],res["ROut"])
	matL = pl.bmat([[ pl.diag(pl.sqrt(P.chiIn)), pl.zeros((M.NM,M.NM))],
						 [ pl.zeros((M.NM,M.NM)), pl.diag(pl.sqrt(P.chiOut)) ]])
	matR = pl.bmat([[ pl.diag(1/pl.sqrt(P.chiIn)), pl.zeros((M.NM,M.NM))],
						 [ pl.zeros((M.NM,M.NM)), pl.diag(1/pl.sqrt(P.chiOut)) ]])
	S2real = matL*S2*matR

	if 0:
		#Define incident wave as a plane wave
		einc = pl.zeros((2*M.NM,1),dtype='complex').view(pl.matrix)
		einc[M.NM//2,0] = 1.

		tmp = S2*einc
		rnew = tmp[:M.NM].view(pl.ndarray).flatten()
		tnew = tmp[M.NM:].view(pl.ndarray).flatten()
		mR,thetaR,Rnew,mT,thetaT,Tnew = calcRT(M,P,rnew,tnew)

		tmp = S2real*einc
		r = tmp[:M.NM].view(pl.ndarray).flatten()
		t = tmp[M.NM:].view(pl.ndarray).flatten()
		idx = P.propModesIn
		R = abs(r[idx])**2
		idx = P.propModesOut
		T =abs(t[idx])**2
		print "-"*50
		print R
		print Rnew
		print T
		print Tnew
		print abs(R-Rnew).max()
		print abs(T-Tnew).max()
		exit()



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
	print(Rres)
	print(Rnew)
	print("-"*10)
	print(Tres)
	print(Tnew)
	print("-"*32)
	print("Error in reflection  : {:.2e}".format(abs(Rnew-Rres).max()))
	print("Error in transmission: {:.2e}".format(abs(Tnew-Tres).max()))
	#print("(note: since the significant numbers are in general between 0.01 and 1")
	#print(" we required a much lower error (1e-6 or better) to confirm that it works)")

	#Define incident wave as a plane wave
	einc = pl.zeros((2*M.NM,1),dtype='complex').view(pl.matrix)
	einc[M.NM//2,0] = 1.

	Stotreal = stackElements(S1real,S2real)
	#Stotreal = S1real
	if 1:
		pl.imshow(abs(S1real),interpolation='none',vmin=0,vmax=1)
		print abs(S1real).max()
		pl.colorbar()
		pl.show()
		pl.imshow(abs(S2real),interpolation='none',vmin=0,vmax=1)
		print abs(S2real).max()
		pl.colorbar()
		pl.show()

	tmp = Stotreal*einc
	r = tmp[:M.NM].view(pl.ndarray).flatten()
	t = tmp[M.NM:].view(pl.ndarray).flatten()

	idx = Pres.propModesIn
	R = abs(r[idx])**2

	idx = Pres.propModesOut
	T =abs(t[idx])**2

	print "-"*50
	print Rres
	print Rnew
	print R
	print("Error in reflection  : {:.2e}".format(abs(R-Rres).max()))
	print("Error in transmission: {:.2e}".format(abs(T-Tres).max()))



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
