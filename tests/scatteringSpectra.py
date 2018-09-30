from __future__ import division
import pylab as pl

from pylab import pi,sin,cos,exp,tanh,sqrt

import os,sys
PATH = os.path.split(os.path.realpath(__file__))[0]+"/../"
sys.path.append(PATH)

import math

import OOPS
#from OOPS.DossouSolver2D.solver import FE,calcRT,calcScatteringMatrix
from OOPS.DossouSolver2D.solverold import FE,calcScatteringMatrix, _printResultsToScreen
from OOPS.DossouSolver2D.physicsClass import physics
from OOPS.DossouSolver2D.modelClass import model
#from auxiliary.dataHandling import loadDicts,saveDicts,dictSlice


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
  for thetadelta in range (0, 120):
   for lambdadelta in range (0, 100):
    calcdp(200+6*lambdadelta,-60+1*thetadelta)
   print("one loop")
   print(thetadelta, lambdadelta)
  print("done")
  return


def calcdp(wavinput,thetainput):
	rep_no = 50
	wav = wavinput
	lam = 400
	thetain = thetainput/180*pl.pi
	nAir = 1.0
	nBac = 1.38
	nMed = 1.34
	#nMaterial2 = nIn	#When nMaterial2 == nIn, the methods works
	#nOut = nMaterial2
	pol = 'Ey'
	nelx = 40
	z_uc = 2.0*math.sqrt(3)

	##Scattering matrix model air-material boundary
	Mres = model(lx=2*lam,lz=(1+rep_no)*(z_uc)*lam,nelx=nelx)
	Pres = physics(Mres,wav,nAir,nBac,thetaIn=thetain,pol=pol)
	if 1:	#Put to zero to avoid re-simulation and use last saved results
		materialResAir = Pres.newMaterial(nAir)
		materialResBac = Pres.newMaterial(nBac)
		materialResMed = Pres.newMaterial(nMed)

		xres = Mres.generateEmptyDesign()
		makeSlab(xres,Mres,(rep_no)*(z_uc)*lam,(1+rep_no)*(z_uc)*lam,materialResAir)	
		makeSlab(xres,Mres,0*(z_uc)*lam,(rep_no)*(z_uc)*lam,materialResMed)	
		
		for x in range (0, rep_no):
			makeCircle(xres,Mres,lam,x*(z_uc)*lam,lam,materialResBac)
			makeCircle(xres,Mres,0,(x+1/2)*(z_uc)*lam,lam,materialResBac)
			makeCircle(xres,Mres,2*lam,(x+1/2)*(z_uc)*lam,lam,materialResBac)
		makeCircle(xres,Mres,lam,(rep_no)*(z_uc)*lam,lam,materialResBac)

		if 0:	#Show design
			pl.imshow(xres.T,interpolation='none',vmin=0,vmax=4)
			pl.colorbar()
			pl.savefig("full_structure.png",bbox_inches='tight')
			pl.show()
			exit()




	##Scattering matrix model air-material boundary
	M = model(lx=2.0*lam,lz=1*(z_uc)*lam,nelx=nelx)
		#Notice how nOut here is nMaterial2
	P = physics(M,wav,nAir,nBac,thetaIn=thetain,pol=pol)
	materialAir = P.newMaterial(nAir)
	materialBac = P.newMaterial(nBac)
	materialMed = P.newMaterial(nMed)
	x = M.generateEmptyDesign()
	makeSlab(x,M,0,1.0*(z_uc)*lam,materialAir)
	makeCircle(x,M,lam,0,lam,materialBac)
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
	
	Stot = S1
	Stotreal = S1real
	thetainNew=thetain

	
	##Scattering matrix model unit cell
	M = model(lx=2.0*lam,lz=(z_uc)*lam,nelx=nelx)
	thetainNew = pl.arcsin(P.nIn/P.nOut*pl.sin(thetainNew))
	P = physics(M,wav,nBac,nBac,thetaIn=thetainNew,pol=pol)#
	materialAir = P.newMaterial(nAir)
	materialBac = P.newMaterial(nBac)
	materialMed = P.newMaterial(nMed)
	x = M.generateEmptyDesign()
	makeSlab(x,M,0,1.0*(z_uc)*lam,materialMed)
	makeCircle(x,M,lam,0,lam,materialBac)
	makeCircle(x,M,lam,1.0*(z_uc)*lam,lam,materialBac)
	makeCircle(x,M,0,(1/2)*(z_uc)*lam,lam,materialBac)
	makeCircle(x,M,2*lam,(1/2)*(z_uc)*lam,lam,materialBac)
	if 0:	
		pl.imshow(x.T,interpolation='none',vmin=0,vmax=4)
		pl.colorbar()
		pl.savefig("lower_half.png",bbox_inches
            ='tight')
		pl.show()
		exit()
	res = calcScatteringMatrix(M,P,x)
	S2 = RTtoS(res["RIn"],res["TIn"],res["TOut"],res["ROut"])
	matL = pl.bmat([[ pl.diag(pl.sqrt(P.chiIn)), pl.zeros((M.NM,M.NM))],
						 [ pl.zeros((M.NM,M.NM)), pl.diag(pl.sqrt(P.chiOut)) ]])
	matR = pl.bmat([[ pl.diag(1/pl.sqrt(P.chiIn)), pl.zeros((M.NM,M.NM))],
						[ pl.zeros((M.NM,M.NM)), pl.diag(1/pl.sqrt(P.chiOut)) ]])
	S2real = matL*S2*matR
	for x in range (1, rep_no+1):
		Stot = stackElements(Stot,S2)
		Stotreal = stackElements(Stotreal,S2real)
		



	RIn,TIn,TOut,ROut = StoRT(Stot)
	results = {}
	results["RIn"] = RIn
	results["TIn"] = TIn
	results["ROut"] = ROut
	results["TOut"] = TOut
	results.update(M.getParameters())
	results.update(P.getParameters())
	#saveDicts("SmatrixRICalculations.h5",results,'a')

	#Define incident wave as a plane wave
	einc = pl.zeros((2*M.NM,1),dtype='complex').view(pl.matrix)
	einc[M.NM//2,0] = 1.
	tmp = Stot*einc
	rnew = tmp[:M.NM].view(pl.ndarray).flatten()
	tnew = tmp[M.NM:].view(pl.ndarray).flatten()
	mR,thetaR,Rnew,mT,thetaT,Tnew = calcRT(Mres,Pres,rnew,tnew)
	for i in range(len(mR)):
	 title = "reflect_mode_n_equals_{:d}_50.csv".format(mR[i])
	 txtdata = open(title,"a")
	 txtdata.write("{:d}, {:f}, {:.8f}\n".format(wavinput,thetainput,Rnew[i]))
	 #print("m={:d}, theta={:f}, mode={:d}, R={:.8f}".format(wavinput,thetainput,mR[i],Rnew[i]))
	 txtdata.close()
	for i in range(len(mT)):
	 title = "trans_mode_n_equals_{:d}_50.csv".format(mT[i])
	 txtdata = open(title,"a")
	 txtdata.write("{:d}, {:f}, {:.8f}\n".format(wavinput,thetainput,Tnew[i]))
	 #print("m={:d}, theta={:f}, mode={:d}, T={:.8f}".format(wavinput,thetainput,mT[i],Tnew[i]))
	 txtdata.close()

	#Define incident wave as a plane wave
	einc = pl.zeros((2*M.NM,1),dtype='complex').view(pl.matrix)
	einc[M.NM//2,0] = 1.
	#Stotreal = S1real
	if 0:
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
	return



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


def makeCircle(x,M,x0,z0,r,mat):
	X,Z = M.getElementCoordinates()
	idx = pl.sqrt((X-x0)**2+(Z-z0)**2) <r
	x[idx] = mat
	return

def makeSlab(x,M,zstart,zstop,mat):
	X,Z = M.getElementCoordinates()
	idx = ((Z-zstart)>=0) * ((Z-zstop)<=0)
	x[idx] = mat
	return



if __name__ == "__main__":
	main()
