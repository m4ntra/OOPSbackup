from __future__ import division,print_function
import pylab as pl

import warnings

from scipy.sparse.linalg import splu

import os,sys
PATH = os.path.split(os.path.realpath(__file__))[0]+"/../"
sys.path.append(PATH)

import DossouSolver2D
#from OOPS.DossouSolver2D.solver import FE,calcRT,calcScatteringMatrix
from DossouSolver2D.matrixAssembly import assembleMvv,assembleMxxfx

def calcScatteringMatrix2(M,P,x):
	## FE analysis to acquire scattering matrices (see Dossou2006) instead of solving for one incident
	##  condition as done in FE().
	#
	#@param M		Model object containing geometrical information
	#@param P		Physics object containing physical parameters
	#@param x		The design domain with 0 corresponding to eps1 and 1 corresponding to eps2
	#						and intermediate values to weighted averages in between. See physics.interpolate
	#@return			Dictionary containing the matrices R,T,R',T' that the scattering matrix consists of.
	#						as well as normal data
	#
	#see also FE()
	A,B = P.interpolate(x)
	Mvv = assembleMvv(A,B,M,P)
	Mrv,Mvr,Mtv,Mvt,fvn,frn = assembleMxxfx(M,P)

	lu = splu(Mvv)
	#Mhatrr and Mhattr
	b = lu.solve(Mvr)
	Mhatrr = -Mrv*b
	_addToDiagonal(Mhatrr, M.lx)
	Mhattr = -Mtv*b

	#Mhatrt and Mhattt
	b = lu.solve(Mvt)
	Mhatrt = -Mrv*b
	Mhattt = -Mtv*b
	_addToDiagonal(Mhattt, M.lx)

	#fhatrn
	b = lu.solve(fvn)
	a = Mrv*b
	fhatrn = frn-a

	#fhattn
	b = lu.solve(fvn)
	a = Mtv*b
	fhattn = -a

	MAT = pl.bmat([[Mhatrr, Mhatrt],
						[Mhattr, Mhattt]])

	RHS = pl.bmat([[ Mhatrr-2*M.lx*pl.identity(M.NM), Mhatrt],
						[Mhattr, Mhattt-2*M.lx*pl.identity(M.NM)]])
	#Solve Eq (53) in Dossou 2006
	RTtilde = pl.solve(MAT,RHS)
	#print("pl.diag(pl.asarray(pl.transpose(RTtilde)).reshape(-1))")
	#print(pl.diag(pl.asarray(pl.transpose(RTtilde)).reshape(-1)))
	matL = pl.bmat([[ pl.diag(pl.sqrt(P.chiIn)), pl.zeros((M.NM,M.NM))],
						 [ pl.zeros((M.NM,M.NM)), pl.diag(pl.sqrt(P.chiOut)) ]])
	matR = pl.bmat([[ pl.diag(1/pl.sqrt(P.chiIn)), pl.zeros((M.NM,M.NM))],
						 [ pl.zeros((M.NM,M.NM)), pl.diag(1/pl.sqrt(P.chiOut)) ]])
	V = matL*RTtilde*matR
  
	RIn = V[:M.NM,:M.NM]
	TIn = V[M.NM:,:M.NM]
	ROut = V[M.NM:,M.NM:]
	TOut = V[:M.NM,M.NM:]

	results = {}
	results["x"] = x
	results["RIn"] = RIn
	results["TIn"] = TIn
	results["ROut"] = ROut
	results["TOut"] = TOut
	results.update(M.getParameters())
	results.update(P.getParameters())
	return results

def combineMatrices(*list):
  Stot =  RTtoS(list[0]["RIn"],list[0]["TIn"],list[0]["TOut"],list[0]["ROut"])
  for i in range(1,len(list)):
    if list[i-1]["nOut"] != list[i]["nIn"]:
      warnings.warn("Boundaries do not match. Results may be flawed.",Warning)
    Stot =  stackElements(Stot, RTtoS(list[i]["RIn"],list[i]["TIn"],list[i]["TOut"],list[i]["ROut"]))
    if 0: #show design
      pl.imshow(list[i]["x"].T,interpolation='none')
      pl.colorbar()
      pl.show()
    return Stot
      
def calcRT(M,P,r,t):
	m = pl.arange(-M.Nm,M.Nm+1)

	idx = P.propModesIn
	mR = m[idx]
	thetaR =  P.thetaModesIn[idx].real
	R = abs(r[idx])**2

	idx = P.propModesOut
	mT = m[idx]
	thetaT = P.thetaModesOut[idx].real
	T = abs(t[idx])**2
	if P.pol == 'Hy':
		T = T**2 

	return mR,thetaR,R,mT,thetaT,T
	


def findSolution(Mres,Pres,Stot):
	#e.g. for normal incident wave
	#Define incident wave as a plane wave
	einc = pl.zeros((2*Mres.NM,1),dtype='complex').view(pl.matrix)
	einc[Mres.NM//2,0] = 1.
	tmp = Stot*einc
	rnew = tmp[:Mres.NM].view(pl.ndarray).flatten()
	tnew = tmp[Mres.NM:].view(pl.ndarray).flatten()
	mR,thetaR,Rnew,mT,thetaT,Tnew = calcRT(Mres,Pres,rnew,tnew)
	return mR,thetaR,Rnew,mT,thetaT,Tnew 


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

def _addToDiagonal(matrix,val):
	matrix.flat[::matrix.shape[0]+1] += val

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
