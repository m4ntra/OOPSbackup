from __future__ import division,print_function

import pylab as pl
from pylab import pi,sin,cos,exp

from scipy.sparse.linalg import splu #sparse LU solver

from matrixAssembly import assembleMvv,assembleMxxfx


def FE(M,P,x,printResults=True):
	## FE analysis to solve Maxwell's equations in 2D for either H or E polarisation (=Helmholtz's equation)
	##		using periodic boundary conditions and wave expansions at the boundaries. The approach was originally
	##		implemented using Fuchi2010 and later modifications from Dossou2006 (original article) were added
	#
	#@param M		Model object containing geometrical information
	#@param P		Physics object containing physical parameters
	#@param x		The design domain with 0 corresponding to eps1 and 1 corresponding to eps2
	#						and intermediate values to weighted averages in between. See physics.interpolate
	#@param printResults	print results to stdout after simulation
	#@return			sol is the discretised solution, r are the complex reflection coefficients
	#						and t are the complex transmission coefficients

	#Calculate A and B, as the would be given in the Helmholtz equation in Eq (1) in Friis2012.
	#	(this notation makes it easy to switch between E and H field due to duality)

	A,B = P.interpolate(x)
	Mvv = assembleMvv(A,B,M,P)
	Mrv,Mvr,Mtv,Mvt,fvn,frn = assembleMxxfx(M,P)

	frn[M.Nm,0] = -M.lx			#mode 0

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
	RHS = pl.bmat([ [fhatrn],[fhattn]])

	V  = pl.solve(MAT,RHS)
	r = V[:M.NM,0]
	t = V[M.NM:,0]

	#Solve the system using LU factorisation (as recommended in Dossou2006, p 130, bottom)
	sol = lu.solve(fvn-Mvr*r-Mvt*t)

	r = r.view(pl.ndarray).ravel()
	t = t.view(pl.ndarray).ravel()
	#Cast solution into a form that matches the input model
	sol = sol.reshape(M.nelx+1,M.nelz+1,order='F')
	#Print simulation results
	if printResults:
		_printResultsToScreen(M,P,r,t)

	results = {}
	results["solution"] = sol
	results["x"] = x
	results["r"] = r
	results["t"] = t
	results.update(M.getParameters())
	results.update(P.getParameters())
	return results


def calcScatteringMatrix(M,P,x):
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
	RIn = RTtilde[:M.NM,:M.NM]
	TIn = RTtilde[M.NM:,:M.NM]
	ROut = RTtilde[M.NM:,M.NM:]
	TOut = RTtilde[:M.NM,M.NM:]

	results = {}
	results["RIn"] = RIn
	results["TIn"] = TIn
	results["ROut"] = ROut
	results["TOut"] = TOut
	results.update(M.getParameters())
	results.update(P.getParameters())
	return results


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

	
def calcscaleRT(M,P,r,t):
	m = pl.arange(-M.Nm,M.Nm+1)

	idx = P.propModesIn
	mR = m[idx]
	thetaR =  P.thetaModesIn[idx].real
	R =   abs(r[idx])**2

	idx = P.propModesOut
	mT = m[idx]
	thetaT = P.thetaModesOut[idx].real
	T = abs(t[idx])**2
	if P.pol == 'Hy':
		#T = T*(P.nIn/P.nOut)**2 
		T = T**2 

	return mR,thetaR,R,mT,thetaT,T
	

def _printResultsToScreen(M,P,r,t):
	mR,thetaR,R,mT,thetaT,T = calcRT(M,P,r,t)
	thetaR[abs(thetaR.imag)>1e-12] = pl.nan
	#Reflection modes
	for i in range(len(mR)):
		print(" m={:>3d}    theta_in={:6.1f}    R={:.3f}".format(mR[i],thetaR[i]/pi*180,R[i]))
	print("Rtot= {:.3f}".format(sum(R)))

	for i in range(len(mT)):
		print(" m={:>3d}    theta_out={:6.1f}    T={:.3f}".format(mT[i],thetaT[i]/pi*180,T[i]))
	print("Ttot= {:.3f}".format(sum(T)))

	print("Ey - Ttot+Rtot= {:.5f}".format(sum(T)+sum(R)))

	return
	
  
def _printResultsToScreenScale(M,P,r,t):
	print("Result of calculated values")
	mR,thetaR,R,mT,thetaT,T = calcscaleRT(M,P,r,t)
	thetaR[abs(thetaR.imag)>1e-12] = pl.nan
	#Reflection modes
	for i in range(len(mR)):
		print(" m={:>3d}    theta_in={:6.1f}    R={:.3f}".format(mR[i],thetaR[i]/pi*180,R[i]))
	print("Rtot= {:.3f}".format(sum(R)))

	for i in range(len(mT)):
		print(" m={:>3d}    theta_out={:6.1f}    T={:.3f}".format(mT[i],thetaT[i]/pi*180,T[i]))
	print("Ttot= {:.3f}".format(sum(T)))

	print("Ey - Ttot+Rtot= {:.5f}".format(sum(T)+sum(R)))

	return
	

def _addToDiagonal(matrix,val):
	matrix.flat[::matrix.shape[0]+1] += val



if __name__ == "__main__":
	pass
