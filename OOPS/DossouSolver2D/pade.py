from __future__ import division,print_function

import pylab as pl
from pylab import pi,sin,cos,exp

from scipy.sparse import coo_matrix, csc_matrix

from scipy.sparse.linalg import splu

from solver import calcRT

#Jensen 2007 5,6,7

BCpen = 1e8	#Should be much higher!!

def pade(N, lam0,dlam, M,P,x):
  N = 2	#The current implementation is only valid for N=2
  #diferenctaite all the elemnts of K by k 
  #Fvec has relation to k_0 through alpha - facter hthat 
  #if possible separte Fvec out into elements of facters of alpha
  #equal to eq 3 in papser, make into coo_matrix for input
  A,B = P.interpolate(x)

  Mvv_0 = assembleMvv(A,B,M,P)
  Mrv_0,Mvr_0,Mtv_0,Mvt_0,fvn_0,frn_0, FFUp_0, FFDown_0 = assembleMxxfx(M,P)

  Mvv_1 = assembledev1Mvv(A,B,M,P)
  Mrv_1,Mvr_1,Mtv_1,Mvt_1,fvn_1,frn_1, FFUp_1, FFDown_1 = assembledev1Mxxfx(M,P, FFUp_0, FFDown_0)

  Mvv_2 = assembledev2Mvv(A,B,M,P)
  #check fvn. frn
  Mrv_2,Mvr_2,Mtv_2,Mvt_2,fvn_2,frn_2 = assembledev2Mxxfx(M,P, FFUp_0, FFDown_0, FFUp_1, FFDown_1)

 # These matrices are just used for checks later on:
  I = pl.identity(M.NM)	#Identity matrix
  Z = pl.zeros((M.NM,M.NM))
  P_0 = pl.bmat([[Mvv_0, Mvr_0, Mvt_0],
 					[Mrv_0, M.lx*I, Z     ],
					[Mtv_0, Z     ,M.lx*I]])
  P_1 = pl.bmat([[Mvv_1, Mvr_1, Mvt_1],
					[Mrv_1, Z      , Z    ],
					[Mtv_1, Z      , Z    ]])
  P_2 = pl.bmat([[Mvv_2, Mvr_2, Mvt_2],
					[Mrv_2, Z      , Z    ],
					[Mtv_2, Z      , Z    ]])
 #Note that u contains the same content as sol,r and t
  u = []
  sol = []
  r = []
  t = []  

  #Note that the LU factorisation on calcvrt0 can be reused for calculating all the u's!
  #(which would speed up things, but not change the result obviously...)
  frn_0[M.Nm,0] = -M.lx			#mode 0 incidence
  ftn_0 = pl.zeros(frn_0.shape, dtype=complex)
  sol_0, r_0, t_0 = calcvrt0(M, Mvv_0, Mrv_0,Mvr_0,Mtv_0,Mvt_0,fvn_0,frn_0, ftn_0) 
  sol.append(sol_0)
  r.append(r_0)
  t.append(t_0)
  u.append(pl.concatenate((sol_0, r_0, t_0)))
  if 1:
    #Check that the solution is correct
    f =  pl.concatenate((fvn_0,frn_0,ftn_0))
    print("Solution error for 0'th order: ",abs(P_0*u[0]-f).max())
  _Mvv = Mvv_0
  _Mrv = Mrv_0
  _Mvr = Mvr_0
  _Mtv = Mtv_0
  _Mvt = Mvt_0
  _fvn = -(Mvv_1*sol[0] + Mvr_1*r[0] + Mvt_1*t[0])
  _frn = -(Mrv_1*sol[0])
  _ftn = -(Mtv_1*sol[0])

  sol_1, r_1, t_1 = calcvrt0(M, _Mvv,_Mrv,_Mvr,_Mtv,_Mvt,_fvn,_frn,_ftn)
  sol.append(sol_1)
  r.append(r_1)
  t.append(t_1)
  u.append(pl.concatenate((sol_1, r_1, t_1)))
  if 1:
    print("Solution error for 1'th order: ",abs(P_0*u[1]+P_1*u[0]).max())

  #This will not work for our implementation since higher order derivates of the matrix is non-zero
  #And eqn(19) in Jensen 2007 therefore has to be recalculated for our specific case. This is caused by
  #derivations of the small dense transmission/reflection matrices (Mrv,Mvt and so on) being non-zero
  if 0:
    for i in range(2,N+1):
      sol_i, r_i, t_i = calcvrt0(M, Mvv_0, Mrv_0,Mvr_0,Mtv_0,Mvt_0,-(Mvv_1*sol[i-1] + Mvr_1*r[i-1] + Mvt_1*t[i-1])-(Mvv_2*sol[i-2] + Mvr_2*r[i-2] + Mvt_2*t[i-2]),-(Mrv_1*sol[i-1])-(Mrv_2*sol[i-2]), -(Mtv_1*sol[i-1])-(Mtv_2*sol[i-2])) 
      sol.append(sol_i)
      r.append(r_i)
      t.append(t_i)
      u.append(pl.concatenate((sol_i, r_i, t_i)))

    sol_N1, r_N1, t_N1 = calcvrt0(M, Mvv_0, Mrv_0,Mvr_0,Mtv_0,Mvt_0,-(Mvv_1*sol[N] + Mvr_1*r[N] + Mvt_1*t[N])-(Mvv_2*sol[N-1] + Mvr_2*r[N-1] + Mvt_2*t[N-1]),-(Mrv_1*sol[N])-(Mrv_2*sol[N-1]), -(Mtv_1*sol[N])-(Mtv_2*sol[N-1])) 

  _fvn = -(Mvv_1*sol[1] + Mvr_1*r[1] + Mvt_1*t[1])
  _fvn += -(Mvv_2*sol[0] + Mvr_2*r[0] + Mvt_2*t[0])
  _frn = -(Mrv_1*sol[1])
  _frn += -(Mrv_2*sol[0])
  _ftn = -(Mtv_1*sol[1])
  _ftn += -(Mtv_2*sol[0])
  sol_2, r_2, t_2 = calcvrt0(M, _Mvv,_Mrv,_Mvr,_Mtv,_Mvt,_fvn,_frn,_ftn)
  sol.append(sol_2)
  r.append(r_2)
  t.append(t_2)
  u.append(pl.concatenate((sol_2, r_2, t_2)))
  if 1:
    print("Solution error for 2'th order: ",abs(P_0*u[2] +P_1*u[1]+P_2*u[0]).max())


  uconjT = pl.bmat([vec.conj() for vec in u[::-2]]).T 
  urev = pl.bmat(u[::-2])
  Pmat = uconjT*urev
  Q = pl.linalg.inv(Pmat)
  #Check the inversion... Even though it succeeds
  if 0:
    print("Pmax:",abs(Pmat).max())
    print("P*Q:")
    print(Pmat*Q)
  uplus = Q*uconjT
  b = -uplus*u[N]
  #Check if b solves the equation we wanted
  if 1:
    print("Error for b:",abs(urev*b+u[N]).max())
	
  a = []
  a += [0]	#If we start summing at 1, then the 0'th element should be zero
  #b starts at index 1 and not zero in Jensen 2007, there the elements have to be shifted one
  a += [u[1]+b[0,0]*u[0]]
  if N != 2:
    exit("this part only implemented for N=2")
  a += [u[2]+b[0,0]*u[1]+b[1,0]*u[0]]


  #List of wavelengths
  llam = pl.linspace(lam0-dlam,lam0+dlam,81)
  #List of sigmas with k0 subtracted
  lsigma = 2*pl.pi/llam-2*pl.pi/lam0
  lR = []
  for sigma in lsigma:
    u_nom = u[0]+a[1]*sigma+a[2]*sigma**2
    u_denom = 1+b[0,0]*sigma+b[1,0]*sigma**2
    usigma = u_nom/u_denom

    nnodes = (M.nelx+1)*(M.nelz+1)
    solsigma = usigma[:nnodes]
    rsigma = usigma[nnodes:nnodes+M.NM]
    tsigma = usigma[nnodes+M.NM:]
    mR,thetaR,R,mT,thetaT,T = calcRT(M,P,rsigma,tsigma)
    lR += [float(R[mR==0])]

  return llam,lR


def assembleMvv(A,B,M,P):
	#A	?
	#B	?
	#M Mesh object
	#P Physics object

	nodes = 4
	sa0,ma0 = localStiffness(M)

	etabflat = M.etab.reshape(M.nelx*M.nelz,nodes)
	Aflat = A.flatten()[pl.newaxis].T
	Bflat = B.flatten()[pl.newaxis].T

	inew = pl.kron(etabflat,pl.ones((1,nodes))).flatten()
	jnew = pl.kron(etabflat,pl.ones((nodes,1))).flatten()
	dnew = pl.dot(Aflat,sa0.reshape((1,nodes*nodes))).flatten() - P.k0**2*pl.dot(Bflat,ma0.reshape((1,nodes*nodes))).flatten()
	i = list(inew)
	j = list(jnew)
	d = list(dnew)

	#Periodic Bloch-Floquet BC:
	pen = BCpen
	n1 = list(range(0,(M.nelz+1)*(M.nelx+1),M.nelx+1))
	n2 = list(range(M.nelx,(M.nelz+1)*(M.nelx+1)+M.nelx,M.nelx+1))
	i += n1+n2+n1+n2
	j += n1+n2+n2+n1
	h =  [pen]*2*len(n1)+\
				[-pen*exp(1.j*P.kInx*M.lx)]*len(n1)+\
				[-pen*exp(1.j*P.kInx*M.lx).conj()]*len(n1)
	d += [pen]*2*len(n1)+\
				[-pen*exp(1.j*P.kInx*M.lx)]*len(n1)+\
				[-pen*exp(1.j*P.kInx*M.lx).conj()]*len(n1)

	#Calculate the matrices given in Eq (43) in Dossou2006. See Fuchi2010 for a "nicer" way
	#	of writing them. The elements used are the same as in Andreassen2011
	
	Mvv = coo_matrix((d,(i,j)),shape=(M.ndof,M.ndof),dtype='complex').toarray()
	return Mvv

def assembledev1Mvv(A,B,M,P):
	#A	?
	#B	?
	#M Mesh object
	#P Physics object

	nodes = 4
	sa0,ma0 = localStiffness(M)

	etabflat = M.etab.reshape(M.nelx*M.nelz,nodes)
	Aflat = A.flatten()[pl.newaxis].T
	Bflat = B.flatten()[pl.newaxis].T

	inew = pl.kron(etabflat,pl.ones((1,nodes))).flatten()
	jnew = pl.kron(etabflat,pl.ones((nodes,1))).flatten()
	#dnew =  pl.dot(Aflat,sa0.reshape((1,nodes*nodes))).flatten() -P.k0**2*pl.dot(Bflat,ma0.reshape((1,nodes*nodes))).flatten()
	dnew = -2*P.k0* pl.dot(Bflat,ma0.reshape((1,nodes*nodes))).flatten()
	i = list(inew)
	j = list(jnew)
	d = list(dnew)
	#Periodic Bloch-Floquet BC:
	pen = BCpen
	n1 = list(range(0,(M.nelz+1)*(M.nelx+1),M.nelx+1))
	n2 = list(range(M.nelx,(M.nelz+1)*(M.nelx+1)+M.nelx,M.nelx+1))
	i += n1+n2+n1+n2
	j += n1+n2+n2+n1
	d += [0]*2*len(n1)+\
				[-pen*1.j*P.kInx*M.lx/P.k0*exp(1.j*P.kInx*M.lx)]*len(n1)+\
				[-pen*1.j*P.kInx*M.lx/P.k0*exp(1.j*P.kInx*M.lx).conj()]*len(n1)

	#Calculate the matrices given in Eq (43) in Dossou2006. See Fuchi2010 for a "nicer" way
	#	of writing them. The elements used are the same as in Andreassen2011
	#Mvv
	Mvv = coo_matrix((d,(i,j)),shape=(M.ndof,M.ndof),dtype='complex').toarray()
	return Mvv

       
def assembledev2Mvv(A,B,M,P):
	#A	?
	#B	?
	#M Mesh object
	#P Physics object

	nodes = 4
	sa0,ma0 = localStiffness(M)

	etabflat = M.etab.reshape(M.nelx*M.nelz,nodes)
	Aflat = A.flatten()[pl.newaxis].T
	Bflat = B.flatten()[pl.newaxis].T

	inew = pl.kron(etabflat,pl.ones((1,nodes))).flatten()
	jnew = pl.kron(etabflat,pl.ones((nodes,1))).flatten()
	dnew = - pl.dot(Bflat,ma0.reshape((1,nodes*nodes))).flatten()
	i = list(inew)
	j = list(jnew)
	d = list(dnew)

	#Periodic Bloch-Floquet BC:
	pen = BCpen
	n1 = list(range(0,(M.nelz+1)*(M.nelx+1),M.nelx+1))
	n2 = list(range(M.nelx,(M.nelz+1)*(M.nelx+1)+M.nelx,M.nelx+1))
	i += n1+n2+n1+n2
	j += n1+n2+n2+n1
	d += [0]*2*len(n1)+[-0.5*pen*1.j*P.kInx*M.lx*1.j*P.kInx*M.lx/P.k0/P.k0*exp(1.j*P.kInx*M.lx)]*len(n1)+[-0.5*pen*1.j*P.kInx*M.lx*1.j*P.kInx*M.lx/P.k0/P.k0*exp(1.j*P.kInx*M.lx).conj()]*len(n1)

	#Calculate the matrices given in Eq (43) in Dossou2006. See Fuchi2010 for a "nicer" way
	#	of writing them. The elements used are the same as in Andreassen2011
	#Mvv
	Mvv = coo_matrix((d,(i,j)),shape=(M.ndof,M.ndof),dtype='complex').toarray()
	return Mvv


#P contains info on material at boundaries
def assembleMxxfx(M,P):
	#Instead of solving Eq (43) in Dossou2006 directly, it is faster to solve the reduced systems
	#	shown in Eq (47-52) in Dossou2006. This is what is done below. These matrices are not banded
	#  and therefore treated as dense matrices

	#Precalculate integrals for contstructing the matrices:
	iFDown, jFDown, dFDown = [], [], []
	iFUp  , jFUp  , dFUp   = [], [], []
	for m in range(-M.Nm,M.Nm+1):
		idx = m+M.Nm
		elx = pl.array(range(M.nelx))
		x1 = elx/M.nelx*M.lx
		x2 = (elx+1)/M.nelx*M.lx
		Fvec = FourierInt(x1,x2,P.alpha[idx])

		#Downstream modes (... or have I switched what's up and down)
		elzDown = 0	
		edofDown = M.etab[elx,elzDown]
		FvecDown = Fvec.copy()
		FvecDown[1:3,:] = 0.
		iFDown += list(edofDown.flatten())
		jFDown += [idx]*edofDown.size
		dFDown += list(FvecDown.T.flatten())

		#Upstream modes
		elzUp = M.nelz-1 
		edofUp = M.etab[elx,elzUp]
		FvecUp = Fvec.copy()
		FvecUp[0,:] = 0.
		FvecUp[3,:] = 0.
		iFUp += list(edofUp.flatten())
		jFUp += [idx]*edofUp.size
		dFUp += list(FvecUp.T.flatten())

	FFDown =	coo_matrix( (dFDown,(iFDown,jFDown)), shape=(M.ndof,M.NM),dtype='complex').toarray()
	FFUp   =	coo_matrix( (dFUp  ,(iFUp  ,jFUp  )), shape=(M.ndof,M.NM),dtype='complex').toarray()

	Mrv = -FFDown.T
	Mvr = 1.j*P.chiIn*P.AIn*FFDown.conj()
	fvn = pl.zeros((M.ndof,1),dtype='complex')
	fvn[:,0] = 1.j*P.chiIn[M.Nm]*P.AIn*FFDown[:,M.Nm].conj()

	Mtv = -FFUp.T
	Mvt = 1.j*P.chiOut*P.AOut*FFUp.conj()
	frn = pl.zeros((M.NM,1),dtype='complex')

	return Mrv.view(pl.matrix),Mvr.view(pl.matrix), \
			 Mtv.view(pl.matrix),Mvt.view(pl.matrix), \
			 fvn,frn, FFUp, FFDown


#P contains info on material at boundaries
def assembledev1Mxxfx(M,P, FFUp_0, FFDown_0):
	#Instead of solving Eq (43) in Dossou2006 directly, it is faster to solve the reduced systems
	#	shown in Eq (47-52) in Dossou2006. This is what is done below. These matrices are not banded
	#  and therefore treated as dense matrices

	#Precalculate integrals for contstructing the matrices:
	iFDown, jFDown, dFDown = [], [], []
	iFUp  , jFUp  , dFUp   = [], [], []
  #-M.Nm,M.Nm+1
	for m in range(-M.Nm,M.Nm+1):
		idx = m+M.Nm
		elx = pl.array(range(M.nelx))
		x1 = elx/M.nelx*M.lx
		x2 = (elx+1)/M.nelx*M.lx
		Fvec = DevFourierInt(x1,x2,P.alpha[idx], P)

		#Downstream modes (... or have I switched what's up and down)
		elzDown = 0	
		edofDown = M.etab[elx,elzDown]
		FvecDown = Fvec.copy()
		FvecDown[1:3,:] = 0.
		iFDown += list(edofDown.flatten())
		jFDown += [idx]*edofDown.size
		dFDown += list(FvecDown.T.flatten())

		#Upstream modes
		elzUp = M.nelz-1 
		edofUp = M.etab[elx,elzUp]
		FvecUp = Fvec.copy()
		FvecUp[0,:] = 0.
		FvecUp[3,:] = 0.
		iFUp += list(edofUp.flatten())
		jFUp += [idx]*edofUp.size
		dFUp += list(FvecUp.T.flatten())

	FFDown =	coo_matrix( (dFDown,(iFDown,jFDown)), shape=(M.ndof,M.NM),dtype='complex').toarray()
	FFUp   =	coo_matrix( (dFUp  ,(iFUp  ,jFUp  )), shape=(M.ndof,M.NM),dtype='complex').toarray()

	Mrv = -FFDown.T
	Mvr = 1.j*P.AIn*(P.chiIn*FFDown.conj()+FFDown_0.conj()*(P.kIn**2/P.k0-P.alpha*P.alpha0/P.k0)/P.chiIn)
	fvn = pl.zeros((M.ndof,1),dtype='complex')
	fvn[:,0] = 1.j*P.AIn*(P.chiIn[M.Nm]*FFDown[:,M.Nm].conj()+(P.kIn**2/P.k0-P.alpha[M.Nm]*P.alpha0/P.k0)/P.chiIn[M.Nm]*FFDown_0[:,M.Nm].conj())

	Mtv = -FFUp.T
	Mvt = 1.j*P.AIn*(P.chiOut*FFUp.conj()+FFUp_0.conj()*(P.kOut**2/P.k0-P.alpha*P.alpha0/P.k0)/P.chiOut)

	frn = pl.zeros((M.NM,1),dtype='complex')

	return Mrv.view(pl.matrix),Mvr.view(pl.matrix), \
			 Mtv.view(pl.matrix),Mvt.view(pl.matrix), \
			 fvn,frn, FFUp, FFDown
       
       

#P contains info on material at boundaries
def assembledev2Mxxfx(M,P, FFUp_0, FFDown_0, FFUp_1, FFDown_1):
	#Instead of solving Eq (43) in Dossou2006 directly, it is faster to solve the reduced systems
	#	shown in Eq (47-52) in Dossou2006. This is what is done below. These matrices are not banded
	#  and therefore treated as dense matrices

	#Precalculate integrals for contstructing the matrices:
	iFDown, jFDown, dFDown = [], [], []
	iFUp  , jFUp  , dFUp   = [], [], []
	for m in range(-M.Nm,M.Nm+1):
		idx = m+M.Nm
		elx = pl.array(range(M.nelx))
		x1 = elx/M.nelx*M.lx
		x2 = (elx+1)/M.nelx*M.lx
		Fvec = Dev2FourierInt(x1,x2,P.alpha[idx], P)

		#Downstream modes (... or have I switched what's up and down)
		elzDown = 0	
		edofDown = M.etab[elx,elzDown]
		FvecDown = Fvec.copy()
		FvecDown[1:3,:] = 0.
		iFDown += list(edofDown.flatten())
		jFDown += [idx]*edofDown.size
		dFDown += list(FvecDown.T.flatten())

		#Upstream modes
		elzUp = M.nelz-1 
		edofUp = M.etab[elx,elzUp]
		FvecUp = Fvec.copy()
		FvecUp[0,:] = 0.
		FvecUp[3,:] = 0.
		iFUp += list(edofUp.flatten())
		jFUp += [idx]*edofUp.size
		dFUp += list(FvecUp.T.flatten())

	FFDown =	coo_matrix( (dFDown,(iFDown,jFDown)), shape=(M.ndof,M.NM),dtype='complex').toarray()
	FFUp   =	coo_matrix( (dFUp  ,(iFUp  ,jFUp  )), shape=(M.ndof,M.NM),dtype='complex').toarray()

	Mrv = -FFDown.T	
	Mvr = 1.j*P.AIn*(P.chiIn*FFDown.conj()+\
                  2*FFDown_1.conj()*(P.kIn**2/P.k0-P.alpha*P.alpha0/P.k0)/P.chiIn+\
                  FFDown_0.conj()*(P.kIn**2/P.k0-P.alpha*P.alpha0/P.k0)*(2*P.kIn**2/P.k0-2*P.alpha0/P.k0*(P.alpha))/(2*P.chiIn**3))

	fvn = pl.zeros((M.ndof,1),dtype='complex')
	fvn[:,0] = 1.j*P.chiIn[M.Nm]*P.AIn*FFDown[:,M.Nm].conj()
	fvn[:,0] = 1.j*P.AIn*(P.chiIn[M.Nm]*FFDown[:,M.Nm].conj()+\
                  2*FFDown_1[:,M.Nm].conj()*(P.kIn**2/P.k0-P.alpha[M.Nm]*P.alpha0/P.k0)/P.chiIn[M.Nm]+\
                  FFDown_0[:,M.Nm].conj()*(P.kIn**2/P.k0-P.alpha[M.Nm]*P.alpha0/P.k0)*(2*P.kIn**2/P.k0-2*P.alpha0/P.k0*(P.alpha[M.Nm]))/(2*P.chiIn[M.Nm]**3))


	Mtv = -FFUp.T
	Mvt = 1.j*P.AIn*(P.chiOut*FFUp.conj()+\
                  2*FFUp_1.conj()*(P.kIn**2/P.k0-P.alpha*P.alpha0/P.k0)/P.chiOut+\
                  FFUp_0.conj()*(P.kIn**2/P.k0-P.alpha*P.alpha0/P.k0)*(2*P.kOut**2/P.k0-2*P.alpha0/P.k0*(P.alpha))/(2*P.chiOut**3))

	frn = pl.zeros((M.NM,1),dtype='complex')

	return Mrv.view(pl.matrix),Mvr.view(pl.matrix), \
			 Mtv.view(pl.matrix),Mvt.view(pl.matrix), \
			 fvn,frn



def localStiffness(M):
	#Jacobians (?) in x and z
	xJac = M.elmsize/2
	zJac = M.elmsize/2

	#Temporary variables
	k1 = (zJac**2+xJac**2)/(xJac*zJac)
	k2 = (zJac**2-2*xJac**2)/(xJac*zJac)
	k3 = (xJac**2-2*zJac**2)/(xJac*zJac)

	sa0 = 1/3*pl.array([
		[ k1  , k2/2,-k1/2, k3/2],
		[ k2/2, k1  , k3/2,-k1/2],
		[-k1/2, k3/2, k1  , k2/2],
		[ k3/2,-k1/2, k2/2, k1  ]])

	ma0 = xJac*zJac/9*pl.array([
		[ 4, 2, 1, 2],
		[ 2, 4, 2, 1],
		[ 1, 2, 4, 2],
		[ 2, 1, 2, 4]])
	return sa0,ma0


def FourierInt(x1,x2,alpha):
	xs = (x2-x1)/2
	xd = (x2+x1)/2
	N = 1 if type(xs) is float else xs.size

	Fvec = pl.zeros((4,N),dtype='complex')
	if abs(alpha)<1e-6:
		Fvec[0,:] = xs
		Fvec[1,:] = xs
		Fvec[2,:] = xs
		Fvec[3,:] = xs
	else:
		Fvec[0,:] = -(-2.j*alpha*xs*exp(-1.j*alpha*(-xd+xs)) - exp(-1.j*alpha*(-xd+xs)) 
						+exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs)
		Fvec[1,:] = -(-2.j*alpha*xs*exp(-1.j*alpha*(-xd+xs)) - exp(-1.j*alpha*(-xd+xs)) 
						+exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs)
		Fvec[2,:] = -( 2.j*alpha*xs*exp( 1.j*alpha*( xd+xs)) + exp(-1.j*alpha*(-xd+xs)) 
						-exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs)
		Fvec[3,:] = -( 2.j*alpha*xs*exp( 1.j*alpha*( xd+xs)) + exp(-1.j*alpha*(-xd+xs)) 
						-exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs)
	return Fvec



def DevFourierInt(x1,x2,alpha, P):
	xs = (x2-x1)/2
	xd = (x2+x1)/2
	N = 1 if type(xs) is float else xs.size

	Fvec = pl.zeros((4,N),dtype='complex')
	if abs(alpha)<1e-6:
		Fvec[0,:] = xs
		Fvec[1,:] = xs
		Fvec[2,:] = xs
		Fvec[3,:] = xs
	else:
		Fvec[0,:] = -((-2*alpha*xs*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) - 2.j*xs*exp(-1.j*alpha*(-xd+xs)) 
						+ 1.j*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) +  1.j*(xd+xs)*exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs) - (-2.j*alpha*xs*exp(-1.j*alpha*(-xd+xs)) - exp(-1.j*alpha*(-xd+xs)) + exp(1.j*alpha*(xd+xs))) /(alpha**3 *xs))* P.alpha0/P.k0
		Fvec[1,:] = -((-2*alpha*xs*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) - 2.j*xs*exp(-1.j*alpha*(-xd+xs)) 
						+ 1.j*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) +  1.j*(xd+xs)*exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs) - (-2.j*alpha*xs*exp(-1.j*alpha*(-xd+xs)) - exp(-1.j*alpha*(-xd+xs)) + exp(1.j*alpha*(xd+xs))) /(alpha**3 *xs))* P.alpha0/P.k0
		Fvec[2,:] = -(( -2*( xd+xs)*alpha*xs*exp( 1.j*alpha*( xd+xs)) + 2.j*xs*exp(1.j*alpha*(xd+xs)) - 1.j*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) 
						- 1.j*(xd+xs)*exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs) - ( 2.j*alpha*xs*exp( 1.j*alpha*( xd+xs)) + exp(-1.j*alpha*(-xd+xs)) 
						-exp(1.j*alpha*(xd+xs))) /(alpha**3 *xs))*P.alpha0/P.k0
		Fvec[3,:] = -(( -2*( xd+xs)*alpha*xs*exp( 1.j*alpha*( xd+xs)) + 2.j*xs*exp(1.j*alpha*(xd+xs)) - 1.j*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) 
						- 1.j*(xd+xs)*exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs) - ( 2.j*alpha*xs*exp( 1.j*alpha*( xd+xs)) + exp(-1.j*alpha*(-xd+xs)) 
						-exp(1.j*alpha*(xd+xs))) /(alpha**3 *xs))*P.alpha0/P.k0
    
	return Fvec



def Dev2FourierInt(x1,x2,alpha, P):
	xs = (x2-x1)/2
	xd = (x2+x1)/2
	N = 1 if type(xs) is float else xs.size

	Fvec = pl.zeros((4,N),dtype='complex')
	if abs(alpha)<1e-6:
		Fvec[0,:] = xs
		Fvec[1,:] = xs
		Fvec[2,:] = xs
		Fvec[3,:] = xs
	else:
		Fvec[0,:] = -0.5*(3*(-2.j*alpha*xs*exp(-1.j*alpha*(-xd+xs)) - exp(-1.j*alpha*(-xd+xs)) +exp(1.j*alpha*(xd+xs)))/(alpha**4 *xs)-2*(-2*alpha*xs*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) - 2.j*xs*exp(-1.j*alpha*(-xd+xs)) 
						+ 1.j*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) +  1.j*(xd+xs)*exp(1.j*alpha*(xd+xs)))/(alpha**3 *xs) + (2.j*alpha*(-xd+xs)**2*xs*exp(-1.j*alpha*(-xd+xs))+(-xd+xs)**2*exp(-1.j*alpha*(-xd+xs))-4*(-xd+xs)*xs*exp(-1.j*alpha*(-xd+xs))-(xd+xs)**2*exp(1.j*alpha*(xd+xs)))/(alpha**2 *2*xs))*(P.alpha0/P.k0)**2
		Fvec[1,:] =  -0.5*(3*(-2.j*alpha*xs*exp(-1.j*alpha*(-xd+xs)) - exp(-1.j*alpha*(-xd+xs)) +exp(1.j*alpha*(xd+xs)))/(alpha**4 *xs)-2*(-2*alpha*xs*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) - 2.j*xs*exp(-1.j*alpha*(-xd+xs)) 
						+ 1.j*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) +  1.j*(xd+xs)*exp(1.j*alpha*(xd+xs)))/(alpha**3 *xs) + (2.j*alpha*(-xd+xs)**2*xs*exp(-1.j*alpha*(-xd+xs))+(-xd+xs)**2*exp(-1.j*alpha*(-xd+xs))-4*(-xd+xs)*xs*exp(-1.j*alpha*(-xd+xs))-(xd+xs)**2*exp(1.j*alpha*(xd+xs)))/(alpha**2 *2*xs))*(P.alpha0/P.k0)**2
		
		Fvec[2,:] =   -0.5*(3*(2.j*alpha*xs*exp(1.j*alpha*(xd+xs)) + exp(-1.j*alpha*(-xd+xs)) - exp(1.j*alpha*(xd+xs)))/(alpha**4 *xs)-2*(( -2*( xd+xs)*alpha*xs*exp( 1.j*alpha*( xd+xs)) + 2.j*xs*exp(1.j*alpha*(xd+xs))- 1.j*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) 
						- 1.j*(xd+xs)*exp(1.j*alpha*(xd+xs))))/(alpha**3 *xs) + (-2.j*alpha*(xd+xs)**2*xs*exp(1.j*alpha*(xd+xs))-(-xd+xs)**2*exp(-1.j*alpha*(-xd+xs))-4*(xd+xs)*xs*exp(1.j*alpha*(xd+xs))+(xd+xs)**2*exp(1.j*alpha*(xd+xs)))/(alpha**2 *2*xs))*(P.alpha0/P.k0)**2
		Fvec[3,:] =    -0.5*(3*(2.j*alpha*xs*exp(1.j*alpha*(xd+xs)) + exp(-1.j*alpha*(-xd+xs)) - exp(1.j*alpha*(xd+xs)))/(alpha**4 *xs)-2*(( -2*( xd+xs)*alpha*xs*exp( 1.j*alpha*( xd+xs)) + 2.j*xs*exp(1.j*alpha*(xd+xs))- 1.j*(-xd+xs)*exp(-1.j*alpha*(-xd+xs)) 
						- 1.j*(xd+xs)*exp(1.j*alpha*(xd+xs))))/(alpha**3 *xs) + (-2.j*alpha*(xd+xs)**2*xs*exp(1.j*alpha*(xd+xs))-(-xd+xs)**2*exp(-1.j*alpha*(-xd+xs))-4*(xd+xs)*xs*exp(1.j*alpha*(xd+xs))+(xd+xs)**2*exp(1.j*alpha*(xd+xs)))/(alpha**2 *2*xs))*(P.alpha0/P.k0)**2
	return Fvec


#diagonal = M.lx for non-derivative solution and 0 for all higher order derivatives (eqn 43 in Dossou 2006)
def calcvrt0(M,Mvv, Mrv,Mvr,Mtv,Mvt,fvn,frn,ftn):
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
	fhattn = ftn-a

	MAT = pl.bmat([[Mhatrr, Mhatrt],
						[Mhattr, Mhattt]])
	RHS = pl.bmat([ [fhatrn],[fhattn]])

	V  = pl.solve(MAT,RHS)
	r = V[:M.NM,0]
	t = V[M.NM:,0]
	sol = lu.solve(fvn-Mvr*r-Mvt*t)
	return sol,r,t
  

    
  
def _addToDiagonal(matrix,val):
	matrix.flat[::matrix.shape[0]+1] += val


  

if __name__ == "__main__":
	pass
